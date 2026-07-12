"""
sort_dataset.py
Splits the raw Kaggle car-angle dataset (folders named by degree: 0, 40, 90...)
into data/train/<angle>/ and data/val/<angle>/ (80-20 split) so it matches
the ImageFolder structure expected by dataset.py.

Run once, from your project root:
    python sort_dataset.py
"""

import os
import shutil
import random

SOURCE_DIR = r"C:\Users\USER\Downloads\Proejcts\dataset"

DEST_TRAIN = "data/train"
DEST_VAL = "data/val"

VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: source folder not found: {SOURCE_DIR}")
        print("Edit SOURCE_DIR at the top of this script to your dataset path.")
        return

    class_folders = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, f))
    ]
    print(f"Found {len(class_folders)} classes: {class_folders}")

    for class_name in class_folders:
        src_class_dir = os.path.join(SOURCE_DIR, class_name)
        images = [
            f for f in os.listdir(src_class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_dest = os.path.join(DEST_TRAIN, class_name)
        val_dest = os.path.join(DEST_VAL, class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(train_dest, img))
        for img in val_images:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(val_dest, img))

        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val")

    print("\nDone. Data is now in data/train/ and data/val/")


if __name__ == "__main__":
    main()