import sys, pathlib, shutil, random
from ultralytics import YOLO

CLASS_NAMES = ["NOMINAL", "CAUTION", "WARNING"]
DATASET_DIR = pathlib.Path("ml_dataset")
YOLO_DIR = pathlib.Path("yolo_cls_dataset")

# 1. Clean and create structure
if YOLO_DIR.exists(): shutil.rmtree(YOLO_DIR)
for split in ["train", "val"]:
    for c in CLASS_NAMES:
        YOLO_DIR.joinpath(split, c).mkdir(parents=True, exist_ok=True)

# 2. Gather all images and labels
all_imgs = list(DATASET_DIR.joinpath("images", "all").glob("*.png"))
random.shuffle(all_imgs)
split_idx = int(len(all_imgs) * 0.8)
train_imgs = all_imgs[:split_idx]
val_imgs = all_imgs[split_idx:]

def process_split(imgs, split_name):
    for ip in imgs:
        lp = DATASET_DIR / "labels" / "all" / (ip.stem + ".txt")
        if not lp.exists(): continue
        try: cid = int(lp.read_text().split()[0])
        except: continue
        cname = CLASS_NAMES[cid]
        shutil.copy2(ip, YOLO_DIR / split_name / cname / ip.name)

process_split(train_imgs, "train")
process_split(val_imgs, "val")

print("Dataset prepared for YOLO Classification.")

# 3. Train YOLO
model = YOLO("yolov8n-cls.pt")
print("Training YOLO...")
model.train(
    data=str(YOLO_DIR.absolute()),
    epochs=8, imgsz=224, batch=32,
    patience=3, amp=True,
    device="mps",
    project="ml_models", name="yolo_fast_train",
    exist_ok=True
)

# 4. Copy best model to latest.pt so ml_trainer_v3.py uses it
best_pt = pathlib.Path("ml_models/yolo_fast_train/weights/best.pt")
if best_pt.exists():
    shutil.copy2(best_pt, "ml_models/latest.pt")
    print("Copied best YOLO model to latest.pt")

