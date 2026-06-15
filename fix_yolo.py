import sys, pathlib, shutil

def fix():
    content = pathlib.Path('ml_trainer_v3.py').read_text()
    yolo_func = """def _train_yolo(version: str, epochs: int, imgsz: int) -> str:
    import shutil
    cls_data = DATASET_DIR / "cls_data"
    if cls_data.exists(): shutil.rmtree(cls_data)
    for split in ["train", "val"]:
        for c in CLASS_NAMES:
            (cls_data / split / c).mkdir(parents=True, exist_ok=True)
            
    for split, img_dir, lbl_dir in [("train", TRAIN_IMG, TRAIN_LBL), ("val", VAL_IMG, VAL_LBL)]:
        for ip in img_dir.glob("*.png"):
            lp = lbl_dir / (ip.stem + ".txt")
            if not lp.exists(): continue
            try: cid = int(lp.read_text().split()[0])
            except: continue
            cname = CLASS_NAMES[cid]
            shutil.copy2(ip, cls_data / split / cname / ip.name)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  🚀 YOLOv8n-cls eğitimi (device={device})...")
    from ultralytics import YOLO
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(cls_data),
        epochs=epochs, imgsz=imgsz, batch=16,
        project=str(MODEL_DIR), name=version,
        exist_ok=True, verbose=True, device=device
    )
    best = MODEL_DIR / version / "weights" / "best.pt"
    return str(best if best.exists() else MODEL_DIR / version / "weights" / "last.pt")"""

    import re
    content = re.sub(r'def _train_yolo\(.*?\):.*?(?=\ndef _train_pytorch)', yolo_func + '\n\n', content, flags=re.DOTALL)
    
    # Enable YOLO again
    main_train = """    # Dataset txt bazlı detection tarzı olduğu için doğrudan kendi PyTorch CNN'imizi (ResNet style) kullanıyoruz.
    print("  [!] Ultralytics atlanıyor → PyTorch CNN aktif")
    try:
        import torch
        model_path = _train_pytorch(epochs, imgsz, version, weights)
    except ImportError:
        raise RuntimeError("PyTorch gerekli: pip install torch torchvision")"""
    
    new_train = """    try:
        model_path = _train_yolo(version, epochs, imgsz)
    except Exception as e:
        print(f"  [!] YOLO Hatası: {e} → PyTorch CNN")
        model_path = _train_pytorch(epochs, imgsz, version, weights)"""
    
    content = content.replace(main_train, new_train)
    pathlib.Path('ml_trainer_v3.py').write_text(content)

fix()
