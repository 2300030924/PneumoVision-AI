# split_dataset.py
import os
import shutil

base_dir = "dataset"

for split in ["train", "val"]:
    pneu_dir = os.path.join(base_dir, split, "PNEUMONIA")
    bacterial_dir = os.path.join(base_dir, split, "BACTERIAL")
    viral_dir = os.path.join(base_dir, split, "VIRAL")

    os.makedirs(bacterial_dir, exist_ok=True)
    os.makedirs(viral_dir, exist_ok=True)

    for filename in os.listdir(pneu_dir):
        src = os.path.join(pneu_dir, filename)
        if "_bacteria_" in filename:
            dst = os.path.join(bacterial_dir, filename)
        elif "_virus_" in filename:
            dst = os.path.join(viral_dir, filename)
        else:
            print(f"Skipping unknown file: {filename}")
            continue
        shutil.move(src, dst)

    print(f"✅ Split complete for {split} set")

# Remove old combined PNEUMONIA folder
for split in ["train", "val"]:
    old_dir = os.path.join(base_dir, split, "PNEUMONIA")
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)
        print(f"🗑 Removed old folder: {old_dir}")
