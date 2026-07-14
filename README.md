# Skin Tone In the Wild Dataset (STW)

This repository contains the processing scripts and data organization structure for our facial skin tone identification dataset. The dataset aggregates several sources to provide a robust collection of full-frame and segmented facial images with corresponding skin tone annotations.

The dataset contains roughly 40k thousand images of 3.5k individuals.

---

⚠️ Work in Progress
This is an open repository. The current state of the code and data does not yet reflect the results presented in the paper.


## 🚀 Setup & Installation

To clone this repo:
```bash
git clone https://github.com/vitorpmh/STW.git
```

Create a venv and install requirements (works with python <= 3.12.11 and cuda 11.8)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📂 Project Structure

To set up the environment, your local `data/` directory should be organized as follows:

```text
data/
├── OpenData/                        # Raw external datasets (CASIA-Face-Africa, CASIA-FaceV5(BMP), data_celeb_a)        
│   ├── CASIA-Face-Africa/   
│   ├── CASIA-FaceV5/
│   └── data_celeb_a/                # CelebA data should be structured
│       └── img_align_celeba/
├── images/                          # Images downloaded from drive (or you could download them all to Open data and process them yourself)      
└── splits/                          # Downloaded train/test split definitions from drive (these will be rewritten)
```
---


### 1. Download Core Files
First, access our [Google Drive Folder](https://drive.google.com/drive/u/0/folders/1jPVDyY0m_WH9VRwS6uaEtLyAhiWKF7ye) and perform the following:
* Download the contents of the `images` folder and place them into your local `data/` directory.
* Download the `splits` folder and place it inside the `data/` directory. The training scripts are pre-configured to look for them there.

### 2. External Data Requirements
Due to licensing, some datasets must be downloaded directly from the providers. Place all of these unzipped files into `data/OpenData/`:

| Dataset | Source | Requirements |
| :--- | :--- | :--- |
| **CasiaFaceAfrica** | <a href="https://www.idealtest.org/#/" target="_blank">IdealTest</a> | Account required |
| **CasiaV5** | <a href="https://www.idealtest.org/#/" target="_blank">IdealTest</a> | Account required |
| **CelebA** | <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"> MMLab </a> | Download "Align&Cropped Images" (`img_align_celeba.zip,identity_CelebA.txt`) |

Ensure taht CelebA is inside `data_celeb_a`. 

### 3. Data Processing
Once the external datasets are in `data/OpenData/`, run:

```bash
chmod +x data/dataset_creation/create_data.sh
./data/dataset_creation/create_data.sh
```

**What this script does:**
1. Generates full-image and segmented facial crops.
2. Populates the `image/` subfolders with the new assets.
3. Refactors the annotation.csv to be consistent with the repo path.
4. Creates individual and images splits.

---

## 🧔 Skin parsing (removing hair from the skin crops)

The segmented crops are built from MediaPipe landmark polygons alone, and those are
purely geometric: the `FACE` oval keeps every pixel inside the face contour, so beard,
eyebrows, glasses and hair falling over the forehead all survive into the "skin only"
images. Being dark, they pull the predicted skin tone toward a darker shade.

`main/utils/skin_parsing.py` intersects the same polygons with a face-parsing model, so
only pixels the model calls skin are kept:

```
FACE oval, minus eyes/mouth/eyebrows   (geometric, as before)
& parser skin mask                     (semantic: which pixels are actually skin)
```

Measured against the current pipeline (mean RGB of the retained skin, `segformer`):

| face | before | after | |
| :--- | :--- | :--- | :--- |
| thick beard + glasses | (119, 80, 69) | (138, 93, 81) | beard and lenses gone |
| clean-shaven, dark skin | (50, 43, 41) | (52, 45, 43) | unchanged |
| clean-shaven, dark skin | (117, 85, 79) | (123, 88, 83) | unchanged |
| clean-shaven, light skin | (217, 164, 134) | (214, 166, 139) | unchanged |

Faces with no hair inside the old mask come out essentially unchanged, which is what
makes this safe to run over the whole dataset.

**It does not remove stubble or light beard.** Face parsers label those as `skin`. The
obvious patch -- find the rest of the beard by how dark or rough it is against the
subject's own skin -- was implemented and measured, and it does not work: clean-shaven
faces with a shadowed jaw score *darker and rougher* in the lower face than genuinely
bearded ones, so every threshold that removed stubble also ate the chin off a
clean-shaven face. Colour cannot tell beard from shade, so no such filter ships. See the
module docstring for the alternative (`BEARD_FREE_REGIONS`, an anatomical cut that
guarantees no stubble but costs the cheeks) and why it is not the default either.

It writes a **new dataset variant** and leaves the existing ones untouched, so the two
can be compared:

```bash
python data/dataset_creation/images/skin_parsed_dataset.py            # -> data/images/all_cropped_skin_parsed_images
python data/dataset_creation/images/skin_parsed_dataset.py --backend mediapipe
```

| Backend | Dependency | Notes |
| :--- | :--- | :--- |
| `segformer` (default) | `transformers` | `jonathandinu/face-parsing` (CelebAMask-HQ). Removes hair, eyebrows, glasses, lips, ears, and a full beard, which the dataset labels as `hair`. |
| `mediapipe` | none (already required) | `selfie_multiclass_256x256`. CPU-fast, but it only knows head hair: it calls beard, eyebrows and glasses face skin, and left the thick-beard case above at (121, 82, 70) — barely moved. Use it only if `transformers` is not an option. |

Weights download on first use into `data/models/` (override with `STW_MODEL_DIR`). The
default backend can be overridden with `STW_SKIN_BACKEND`.

To use it directly:

```python
from main.utils import skin_parsing as sp
sp.prefetch()                                            # fetch weights up front
skin = sp.segment_skin(img_rgb, backend="segformer", crop=True)
```

### Running it in a notebook (Colab)

The `segformer` checkpoint is a **323 MB** SegFormer-B5. On an ephemeral runtime the
default HuggingFace cache is wiped between sessions, so it is re-downloaded every time
— unauthenticated, throttled, and slow enough that a run looks like it has hung at
`model.safetensors`. Point the cache at your mounted Drive and it downloads once:

```python
import os
from google.colab import drive
drive.mount('/content/drive')

os.environ['STW_MODEL_DIR'] = '/content/drive/MyDrive/stw_models'   # persists across sessions
os.environ['HF_TOKEN'] = '...'                                      # optional: lifts the rate limit

from main.utils import skin_parsing as sp
sp.prefetch()          # run this in its own cell, before any loop over images
```

`prefetch()` exists so the download is an explicit step rather than something that
happens silently inside the first iteration of a 40k-image loop, where it is
indistinguishable from a hang. If the download does stall, interrupt and re-run it —
HuggingFace resumes rather than restarting.

If you cannot afford the 323 MB at all, `--backend mediapipe` needs only a 250 KB
tflite — but see the table above for what it gives up.

⚠️ If you regenerate the dataset with this, retrain before you infer with it — a model
trained on beard-included crops and served beard-removed crops is a train/serve mismatch.

---

## 🛠️ Usage

Training script and other stuff will come out soon.




## Annotations.

If you'd like to try, you can rewrite the scripts under the `annotation` folder to annotate your own data.


## LICENSE
We are still deriving a specific license. Currently you should follow each dataset LICENSE. 

We also ask for citations:
```
@misc{matias2026largescaledatasetbenchmarkskin,
      title={Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild}, 
      author={Vitor Pereira Matias and Márcus Vinícius Lobo Costa and João Batista Neto and Tiago Novello de Brito},
      year={2026},
      url={https://arxiv.org/abs/2603.02475}, 
}
```
