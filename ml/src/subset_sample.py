import json
import os
import random
import requests
from tqdm import tqdm
from collections import defaultdict

# ===== CONFIG =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'community_fish_detection_dataset.json'))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'images', 'detection'))
N_SAMPLES = 10000                             # total images to sample
BASE_URL = "https://storage.googleapis.com/public-datasets-lila/community-fish-detection-dataset/"
TRAIN_RATIO = 0.8                             # 80% train, 20% val
# ==================

# Create output directories
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load JSON metadata
with open(JSON_PATH, "r") as f:
    metadata = json.load(f)

images = metadata["images"]

# Group images by dataset
groups = defaultdict(list)
for img in images:
    groups[img["dataset"]].append(img)

# Decide how many per dataset (equal split)
datasets = list(groups.keys())
per_dataset = N_SAMPLES // len(datasets)

print(f"Sampling {per_dataset} images per dataset across {len(datasets)} datasets.")

subset = []
for ds, imgs in groups.items():
    if len(imgs) <= per_dataset:
        chosen = imgs  # take all if dataset is small
    else:
        chosen = random.sample(imgs, per_dataset)
    subset.extend(chosen)

print(f"Total sampled images: {len(subset)}")

# Split into train/val respecting is_train flag
train_imgs = [img for img in subset if img.get("is_train", True)]
val_imgs = [img for img in subset if not img.get("is_train", False)]

# If too imbalanced, manually split
if len(train_imgs) < int(TRAIN_RATIO * len(subset)):
    # fill train with extra from val pool
    extra = random.sample(val_imgs, int(TRAIN_RATIO * len(subset)) - len(train_imgs))
    train_imgs.extend(extra)
    val_imgs = [img for img in val_imgs if img not in extra]

print(f"Train set: {len(train_imgs)}, Val set: {len(val_imgs)}")

# Download function
def download_images(imgs, outdir):
    for img in tqdm(imgs):
        url = BASE_URL + img["file_name"]
        fname = os.path.join(outdir, os.path.basename(img["file_name"]))

        if os.path.exists(fname):
            continue

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(fname, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"❌ Failed {url}: {e}")

# Download stratified subset
print("Downloading training set...")
download_images(train_imgs, train_dir)

print("Downloading validation set...")
download_images(val_imgs, val_dir)

print("✅ Stratified download complete.")
