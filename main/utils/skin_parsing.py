"""Semantic skin segmentation, applied on top of the MediaPipe landmark polygons.

The polygons in `mediapipe_vitor` are purely geometric: the FACE oval keeps every
pixel inside the face contour, so beard, moustache, eyebrows, glasses and hair
falling over the forehead all survive into the "skin only" crops. Being dark, they
drag the predicted skin tone toward a darker shade.

So the same landmark polygons are kept, and a segmentation model is intersected with
them to decide which of those pixels are actually skin:

    FACE oval                (geometric, unchanged from skin_only_dataset)
  - eyes / mouth / eyebrows  (geometric)
  & parser skin mask         (semantic: a model says which pixels are skin)

The parser removes hair over the forehead, eyebrows, glasses, lips, ears -- and a
full beard, which CelebAMask-HQ labels as `hair`. Faces with no hair inside the old
mask come out essentially unchanged, which is what makes this safe to apply to the
whole dataset.

What this does NOT remove: stubble and light beard
--------------------------------------------------
Face parsers label those as `skin`, so they survive. The obvious patch -- find the
remaining beard by how dark, desaturated or rough it is against the subject's own
skin -- was implemented and then measured against clean-shaven controls, and it does
not work. The beard and shadow populations overlap completely: clean-shaven faces
with a shadowed jaw routinely score *darker and rougher* in the lower face than
genuinely bearded ones. Every threshold that removed stubble also ate the chin off a
clean-shaven face. Colour cannot tell beard from shade, so no colour filter ships
here.

The anatomical alternative -- keep only the region above the jaw, where hair cannot
grow (`BEARD_FREE_REGIONS`) -- does guarantee no stubble, but it drops the cheeks
along with the beard and leaves the mean riding on the forehead and the specular
highlight on the nose. Measured, that moved dark-skinned subjects up to 70 RGB levels
lighter, trading a beard bias for a worse one. It is available, but it is not the
default.

Backends
--------
segformer : jonathandinu/face-parsing (CelebAMask-HQ, 19 classes). The default.
            Removes hair, brows, glasses, lips and ears semantically. Needs
            `transformers`; imported lazily.
mediapipe : selfie_multiclass_256x256. No extra dependency and CPU-fast, but it
            only knows head hair -- it labels brows and glasses as face skin, so
            those fall to the landmark polygons alone. Use it when installing
            transformers is not an option.
"""

import os
import urllib.request

import cv2 as cv
import numpy as np

from . import mediapipe_vitor as mpv

# selfie_multiclass categories, in label order.
_MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
_MP_FACE_SKIN = 3  # 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others

_SEGFORMER_REPO = "jonathandinu/face-parsing"
_SEGFORMER_KEEP = ("skin", "nose")  # matched against config.id2label, not raw indices

DEFAULT_BACKEND = os.environ.get("STW_SKIN_BACKEND", "segformer")

_BACKEND_CACHE = {}


def model_dir():
    """Where downloaded weights live. Under data/ so .gitignore already covers it."""
    path = os.environ.get(
        "STW_MODEL_DIR",
        os.path.join(os.getcwd(), "data", "models"),
    )
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Backends: image -> boolean "this pixel is facial skin" mask                  #
# --------------------------------------------------------------------------- #

def _mediapipe_segmenter():
    if "mediapipe" not in _BACKEND_CACHE:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        weights = os.path.join(model_dir(), "selfie_multiclass_256x256.tflite")
        if not os.path.exists(weights):
            print(f"Downloading selfie_multiclass weights -> {weights}")
            urllib.request.urlretrieve(_MP_MODEL_URL, weights)

        # Pass the bytes, not the path: on Windows MediaPipe resolves an absolute
        # model_asset_path relative to site-packages and fails to open it.
        with open(weights, "rb") as fh:
            buffer = fh.read()

        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=buffer),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        _BACKEND_CACHE["mediapipe"] = vision.ImageSegmenter.create_from_options(options)
    return _BACKEND_CACHE["mediapipe"]


def _mediapipe_skin_mask(img_rgb):
    import mediapipe as mp

    segmenter = _mediapipe_segmenter()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb)
    )
    result = segmenter.segment(mp_image)
    return result.category_mask.numpy_view() == _MP_FACE_SKIN


def _segformer_model():
    if "segformer" not in _BACKEND_CACHE:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        # Cache into model_dir() rather than the default ~/.cache/huggingface. The
        # checkpoint is a 323 MB SegFormer-B5, and on an ephemeral runtime (Colab)
        # the default cache is wiped between sessions, so it gets re-downloaded
        # every time -- unauthenticated, and slow enough to look like a hang. Point
        # STW_MODEL_DIR at persistent storage (a mounted Drive) and it downloads once.
        cache = model_dir()
        processor = SegformerImageProcessor.from_pretrained(_SEGFORMER_REPO, cache_dir=cache)
        model = SegformerForSemanticSegmentation.from_pretrained(_SEGFORMER_REPO, cache_dir=cache)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _BACKEND_CACHE["segformer"] = (processor, model, device)
    return _BACKEND_CACHE["segformer"]


def _segformer_skin_mask(img_rgb):
    import torch

    processor, model, device = _segformer_model()
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    labels = logits.argmax(dim=1)[0].cpu().numpy()

    # Resolve class ids by name so a different checkpoint ordering cannot silently
    # keep the wrong classes.
    keep_ids = [
        int(i) for i, name in model.config.id2label.items() if name in _SEGFORMER_KEEP
    ]
    return np.isin(labels, keep_ids)


_BACKENDS = {
    "mediapipe": _mediapipe_skin_mask,
    "segformer": _segformer_skin_mask,
}


def parser_skin_mask(img_rgb, backend=DEFAULT_BACKEND):
    """Boolean mask of pixels the segmentation model considers facial skin."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}, expected one of {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend](img_rgb)


def prefetch(backend=DEFAULT_BACKEND):
    """Download and load the backend's weights now.

    Otherwise the first image of a 40k-image run pays for the download, which looks
    like the loop has hung rather than like a fetch in progress. Call it once, up
    front, in a notebook cell of its own.
    """
    print(f"Fetching {backend} weights into {model_dir()} ...")
    if backend == "segformer":
        _segformer_model()
    elif backend == "mediapipe":
        _mediapipe_segmenter()
    else:
        raise ValueError(
            f"unknown backend {backend!r}, expected one of {sorted(_BACKENDS)}"
        )
    print(f"{backend} ready.")


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #

def polygon_mask(shape, regions, info):
    """Union of landmark polygons, rasterised to a boolean mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for region in regions:
        points = np.array([[info[i][3], info[i][4]] for i in region], dtype=np.int32)
        cv.fillPoly(mask, [points], 1)
    return mask.astype(bool)


# The default region: the whole face oval, as in the original skin-only dataset.
# The parser, not the geometry, is what takes the hair out of it.
FACE_REGIONS = [mpv.FACE]

# Non-skin things the parser may not have a class for. Eyebrows are here because
# the mediapipe backend calls them face skin; the eyes because an iris is not skin.
NON_SKIN_REGIONS = [mpv.R_EYE, mpv.L_EYE, mpv.MOUTH, mpv.L_EYEBROW, mpv.R_EYEBROW]

# Opt-in alternative: the beard-free anatomy only -- this repo's mid-face band plus
# the forehead. It cannot contain beard at all, but it also drops the jaw and most
# of the cheek, leaving the mean dominated by the forehead and the specular highlight
# on the nose. On the faces measured here that pushed dark-skinned subjects up to 70
# RGB levels lighter, so it is not the default. Pass it explicitly, and only if you
# need a hard guarantee that no stubble survives.
BEARD_FREE_REGIONS = [mpv.FACE_WITHOUT_BEARD, mpv.forehead_contour]


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def skin_mask(
    img_rgb,
    backend=DEFAULT_BACKEND,
    regions_to_segment=None,
    regions_to_remove=None,
    info=None,
):
    """Beard-free landmark polygons AND parser skin.

    The parser runs on the *unmasked* image on purpose. Feeding it the blacked-out
    polygon output would put a hard artificial edge through the face and degrade the
    segmentation; intersecting two independently computed masks gives the same
    region with a cleaner boundary.

    Args:
        img_rgb: uint8 RGB image containing one face.
        backend: 'segformer' or 'mediapipe'.
        regions_to_segment: landmark polygons to keep. Defaults to the beard-free
            anatomy: the mid-face band plus the forehead.
        regions_to_remove: landmark polygons to drop. Defaults to eyes and eyebrows.
        info: precomputed landmarks from `mediapipe_vitor.info_gen`, if you have them.

    Returns:
        Boolean mask, same height/width as img_rgb.
    """
    if regions_to_segment is None:
        regions_to_segment = FACE_REGIONS
    if regions_to_remove is None:
        regions_to_remove = NON_SKIN_REGIONS
    if info is None:
        info = mpv.info_gen(img_rgb)

    mask = polygon_mask(img_rgb.shape, regions_to_segment, info)
    if regions_to_remove:
        mask &= ~polygon_mask(img_rgb.shape, regions_to_remove, info)

    mask &= parser_skin_mask(img_rgb, backend=backend)
    return mask


def segment_skin(
    img_rgb,
    backend=DEFAULT_BACKEND,
    regions_to_segment=None,
    regions_to_remove=None,
    crop=True,
    return_mask=False,
):
    """Skin-only image: everything that is not facial skin is zeroed out.

    Drop-in replacement for `mediapipe_vitor.show_or_remove_roi` in the skin-tone
    pipeline, with the semantic pass added and the beard anatomy excluded.

    Returns the segmented image, or (image, mask) when return_mask is set.
    """
    mask = skin_mask(
        img_rgb,
        backend=backend,
        regions_to_segment=regions_to_segment,
        regions_to_remove=regions_to_remove,
    )

    result = img_rgb * mask[..., None]
    if crop:
        result = mpv.crop_image(result)

    if return_mask:
        return result, mask
    return result
