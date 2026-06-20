import os
import re
import cv2
import numpy as np

from config import DATASET_PATH, SHAPES, IMG_SIZE, SKIP_FILES


def get_score_from_filename(filename, code):
    pattern = rf"-{re.escape(code)}-(\d+)"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_shape_folder(shape_name):
    expected = shape_name.lower().replace("_", " ")
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder.lower() == expected:
            return folder_path
    raise FileNotFoundError(
        f"Could not find folder for shape '{shape_name}'"
    )


def load_shape(shape_name, img_size=IMG_SIZE):
    shape_info = SHAPES[shape_name]
    code = shape_info["code"]
    folder = find_shape_folder(shape_name)

    images = []
    scores = []
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp")

    for fname in sorted(os.listdir(folder)):
        if fname in SKIP_FILES:
            continue
        if not fname.lower().endswith(valid_ext):
            continue

        score = get_score_from_filename(fname, code)
        if score is None:
            print(f"Skipped (cannot parse score): {fname}")
            continue

        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        images.append(img)
        scores.append(score)

    print(f"{shape_name}: loaded {len(images)} images")
    return images, scores