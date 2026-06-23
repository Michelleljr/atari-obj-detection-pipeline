import os
import shutil
import random
from pathlib import Path

img_dir = Path(r"praktikum_obj_detection_team\data\pong\captured_100frames_pong") #replace with your path of captured frames dir (img+txt)
out_dir = Path("dataset")
val_split = 0.15 

random.seed(42)

for split in ("train", "val"):
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Collect every frame that has BOTH a png and a non-empty txt label file
pairs = []
missing_txt = 0
empty_txt = 0
found_class_ids = set()

all_pngs = sorted(img_dir.glob("*.png"))
print(f"  Found {len(all_pngs)} .png files in {img_dir}")

# Check all images for it's corresponding txt pair
for img_path in all_pngs:
    lbl_path = img_path.with_suffix(".txt")
    if not lbl_path.exists():   # skip the image if no txt file exists
        missing_txt += 1
        continue
    text = lbl_path.read_text(encoding="utf-8-sig").strip()
    if text == "":    # skip the image if the txt file is empty
        empty_txt += 1
        continue
    for line in text.splitlines():
        parts = line.strip().split()
        if parts:
            found_class_ids.add(int(parts[0]))
    pairs.append((img_path, lbl_path))
print("found_class_ids", found_class_ids)
# map found object ids in txt to 0-indexed for training, 
sorted_ids  = sorted(found_class_ids)
remap       = {old: new for new, old in enumerate(sorted_ids)}
print(f"  Remap: {remap}")
print()
print("Paste this into dataset.yaml:")
print("names:")
for old_id, new_id in remap.items():
    print(f"{new_id}: class_{old_id}   # replace with real name")

random.shuffle(pairs)

# Split into train/val
val = max(1, int(len(pairs) * val_split))
train = len(pairs) - val

val_pairs = pairs[:val]
train_pairs = pairs[val:]

# Copy the pairs into the output dataset directory, remapping class IDs in the txt files
def copy_pairs(pair_list, split):
    for img_path, lbl_path in pair_list:
        shutil.copy(img_path, out_dir / "images" / split / img_path.name)
        text = lbl_path.read_text(encoding="utf-8-sig").strip()
        new_lines = []
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            new_id = remap.get(old_id, old_id)
            new_lines.append(f"{new_id} " + " ".join(parts[1:]))
        out_lbl = out_dir / "labels" / split / lbl_path.name
        out_lbl.write_text("\n".join(new_lines))

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs,   "val")

print("─" * 50)
print(f"  Total pairs : {len(pairs)}")
print(f"  Train       : {train}")
print(f"  Val         : {val}")
print(f"  Output      : {out_dir.resolve()}")
print("─" * 50)
print("Dataset is ready for training")