"""Skin-only crops with the hair parsed out of them.

Same region as skin_only_dataset.py, but the geometric landmark polygon is
intersected with a face-parsing model (see main/utils/skin_parsing.py). The polygon
alone keeps beard, eyebrows, glasses and forehead hair inside the FACE oval, and
those dark pixels bias the predicted skin tone downward.

Removes: hair, eyebrows, glasses, lips, ears, and full beards.
Does not remove: stubble and light beard -- parsers label them `skin`, and no colour
filter can separate them from shadow. See the skin_parsing module docstring.

Writes a new variant and leaves the existing datasets alone, so the two can be
compared.

Usage:
    python data/dataset_creation/images/skin_parsed_dataset.py [--backend segformer]
"""

import argparse
import os
import sys

import cv2 as cv
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), '.'))
from main.utils import skin_parsing as sp


output_path = os.path.join('data/images/all_cropped_skin_parsed_images')
os.makedirs(output_path, exist_ok=True)
for i in range(1, 11):
    os.makedirs(os.path.join(output_path, f"{i}"), exist_ok=True)
input_path = 'data/OpenData/'
log = open('data/images/log_skin_parsed.txt', 'w')

paths = []


def process_image(path, label, new_token, backend):
    path_2_save = (output_path
                   + f'/{label}/'
                   + new_token
                   + '_'
                   + path.split('/')[-1])

    # if the path exists with png or other just rewrite it into jpg
    if os.path.exists(path_2_save) and not path_2_save.lower().endswith('.jpg'):
        cv.imwrite(p := path_2_save.rsplit('.', 1)[0] + '.jpg', cv.imread(path_2_save))
        os.remove(path_2_save)
        paths.append(p)
        return
    # replace extension with jpg
    path_2_save = path_2_save.rsplit('.', 1)[0] + '.jpg'
    paths.append(path_2_save)
    if not os.path.exists(path_2_save):
        try:
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = sp.segment_skin(img, backend=backend, crop=True)

            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imwrite(path_2_save, img)
            log.write(f"Processed: {path}\n")
        except Exception as e:
            # No face found, unreadable file, or a degenerate mask.
            log.write(f"Failed to process: {path} ({type(e).__name__}: {e})\n")
    else:
        log.write(f"Already processed: {path}\n")


def copy_image(df, backend):
    for path, label, new_token in tqdm(df[['paths', 'class', 'new_tokens']].values):
        log.write(f"Processing: {path}\n")
        path = os.path.join(input_path, path)
        process_image(path, label, new_token, backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--backend',
        default=sp.DEFAULT_BACKEND,
        choices=['segformer', 'mediapipe'],
        help="segformer (default, needs transformers) or mediapipe (no extra deps)",
    )
    args = parser.parse_args()

    anotations = pd.read_csv(
        'data/splits/updated_annotations.csv',
        dtype={
            'tokens': str,
            'new_tokens': str
        })

    copy_image(anotations, args.backend)
    anotations_skin_parsed = anotations.copy(deep=True)
    anotations_skin_parsed.insert(1, 'paths_skin_parsed', paths)
    anotations_skin_parsed.to_csv('data/splits/anotations_skin_parsed.csv', index=False)
