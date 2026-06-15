"""
TUSAŞ TestLab — ML Trainer v2
==============================
Gerçek YOLO eğitim pipeline'ı.

Özellikler:
  • Train/Val split (%80/%20)
  • Augmentation (flip, rotate, brightness, noise)
  • Erken durdurma (early stopping)
  • Model versioning (her eğitim tarih damgalı)
  • Eğitim sonunda otomatik rapor açma
  • Confusion matrix + precision/recall metrikleri
  • Anomaly detection (outlier skoru)
  • Live training dashboard (terminalde progress bar)

Kullanım:
  python ml_trainer_v2.py summary
  python ml_trainer_v2.py train [--epochs 50] [--imgsz 416]
  python ml_trainer_v2.py predict screenshot.png
  python ml_trainer_v2.py benchmark          # tüm test set üzerinde toplu değerlendirme
  python ml_trainer_v2.py versions           # eğitilmiş model sürümlerini listele
"""

import os
import sys
import json
import time
import shutil
import random
import struct
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# ─── KLASÖR YAPISI ────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "ml_dataset"
IMAGES_DIR  = DATASET_DIR / "images" / "all"
LABELS_DIR  = DATASET_DIR / "labels" / "all"
TRAIN_IMG   = DATASET_DIR / "images" / "train"
TRAIN_LBL   = DATASET_DIR / "labels" / "train"
VAL_IMG     = DATASET_DIR / "images" / "val"
VAL_LBL     = DATASET_DIR / "labels" / "val"
MODEL_DIR   = BASE_DIR / "ml_models"
LOG_PATH    = DATASET_DIR / "collection_log.json"
METRICS_DIR = BASE_DIR / "training_metrics"

for d in (IMAGES_DIR, LABELS_DIR, TRAIN_IMG, TRAIN_LBL,
          VAL_IMG, VAL_LBL, MODEL_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── SINIF HARİTASI ──────────────────────────────────────────────────────────

CLASS_MAP = {
    "NOMINAL":  0,
    "ADVISORY": 0,
    "CAUTION":  1,
    "WARNING":  2,
}
CLASS_NAMES = ["NOMINAL", "CAUTION", "WARNING"]

# WCA panelinin normalize YOLO koordinatları (1600×860 baz)
WCA_BOX = (
    (1200 + 395/2) / 1600,
    (680  + 175/2) / 860,
    395 / 1600,
    175 / 860,
)

# ─── VERİ TOPLAMA ─────────────────────────────────────────────────────────────

@dataclass
class CollectionEntry:
    scenario_id: str
    severity: str
    class_id: int
    image_path: str
    label_path: str
    timestamp: str


def collect_training_data(
    screenshot_path: str,
    scenario_id: str,
    severity: str,
) -> Optional[CollectionEntry]:
    """
    Screenshot'ı YOLO eğitim datasına ekle.
    Augmentation: 3 ekstra kopya oluşturur → her senaryo 4 örnek üretir.
    """
    if not os.path.exists(screenshot_path):
        print(f"   [ML] Screenshot bulunamadı: {screenshot_path}")
        return None

    class_id = CLASS_MAP.get(severity.upper(), 0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Orijinal + 3 augmented kopya
    entries = []
    for aug_idx in range(4):
        stem = f"{scenario_id}_{ts}_aug{aug_idx}"
        dst_img = IMAGES_DIR / f"{stem}.png"
        dst_lbl = LABELS_DIR / f"{stem}.txt"

        if aug_idx == 0:
            shutil.copy2(screenshot_path, dst_img)
        else:
            _augment_image(screenshot_path, dst_img, aug_idx)

        cx, cy, bw, bh = WCA_BOX
        dst_lbl.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        entries.append(CollectionEntry(
            scenario_id=scenario_id,
            severity=severity,
            class_id=class_id,
            image_path=str(dst_img),
            label_path=str(dst_lbl),
            timestamp=datetime.now().isoformat(),
        ))

    log = _load_log()
    log.extend([asdict(e) for e in entries])
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    counts = _count_classes(log)
    print(f"   [ML] +4 örnek ({severity}) | Toplam: "
          f"NOM={counts[0]} CAU={counts[1]} WARN={counts[2]}")
    return entries[0]


def _augment_image(src_path: str, dst_path: str, aug_idx: int):
    """
    Hafif augmentation: brightness, slight crop, horizontal flip.
    numpy/cv2 varsa kullanır; yoksa ham PNG byte manipülasyonu.
    """
    try:
        import numpy as np
        try:
            import cv2
            img = cv2.imread(src_path)
            if img is None:
                shutil.copy2(src_path, dst_path)
                return
            if aug_idx == 1:
                # Brightness +20
                img = np.clip(img.astype(np.int16) + 20, 0, 255).astype(np.uint8)
            elif aug_idx == 2:
                # Brightness -15
                img = np.clip(img.astype(np.int16) - 15, 0, 255).astype(np.uint8)
            elif aug_idx == 3:
                # Horizontal flip
                img = cv2.flip(img, 1)
            cv2.imwrite(str(dst_path), img)
            return
        except (ImportError, OSError):
            pass
        # cv2 yoksa basit numpy manipülasyonu
        img = _read_png_numpy(src_path)
        if img is None:
            shutil.copy2(src_path, dst_path)
            return
        if aug_idx == 1:
            img = np.clip(img.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        elif aug_idx == 2:
            img = np.clip(img.astype(np.int16) - 15, 0, 255).astype(np.uint8)
        elif aug_idx == 3:
            img = img[:, ::-1, :]
        _write_png_numpy(img, dst_path)
    except Exception:
        shutil.copy2(src_path, dst_path)


def _read_png_numpy(path):
    """Minimal PNG reader using numpy only."""
    try:
        import numpy as np
        with open(path, 'rb') as f:
            data = f.read()
        # Find IHDR
        idx = 8
        chunks = {}
        while idx < len(data):
            length = struct.unpack('>I', data[idx:idx+4])[0]
            chunk_type = data[idx+4:idx+8].decode('ascii', errors='ignore')
            chunk_data = data[idx+8:idx+8+length]
            chunks[chunk_type] = chunk_data
            idx += 12 + length
            if chunk_type == 'IEND':
                break
        if 'IHDR' not in chunks:
            return None
        w, h = struct.unpack('>II', chunks['IHDR'][:8])
        # Decompress
        raw = zlib.decompress(b''.join(
            chunks.get('IDAT', b'') if isinstance(chunks.get('IDAT'), bytes) else b''
        ))
        # Reconstruct scanlines (simplified — no filter decoding)
        row_len = w * 3 + 1
        rows = []
        for y in range(h):
            rows.append(raw[y*row_len+1:(y+1)*row_len])
        arr = np.frombuffer(b''.join(rows), dtype=np.uint8).reshape(h, w, 3)
        return arr
    except Exception:
        return None


def _write_png_numpy(arr, path):
    """Very minimal PNG writer."""
    try:
        import numpy as np
        h, w = arr.shape[:2]
        raw_rows = []
        for row in arr:
            raw_rows.append(b'\x00' + row.tobytes())
        raw = b''.join(raw_rows)
        compressed = zlib.compress(raw, 6)
        ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        def chunk(name, data):
            c = name + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', ihdr))
            f.write(chunk(b'IDAT', compressed))
            f.write(chunk(b'IEND', b''))
    except Exception:
        shutil.copy2(str(path).replace('.png', '_orig.png') if '_orig' not in str(path) else str(path), path)


# ─── TRAIN/VAL SPLIT ──────────────────────────────────────────────────────────

def prepare_splits(val_ratio: float = 0.2, seed: int = 42) -> Tuple[int, int]:
    """
    Tüm images/all/* dosyalarını train (%80) ve val (%20) olarak böl.
    Her çalıştırmada yeniden bölümleme yapar (shuffle + seed).
    """
    # Temizle
    for d in (TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL):
        for f in d.glob("*"):
            f.unlink()

    all_images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
    if not all_images:
        return 0, 0

    random.seed(seed)
    random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_ratio))
    val_set   = set(all_images[:n_val])
    train_set = set(all_images[n_val:])

    def link_pair(img_path: Path, img_dir: Path, lbl_dir: Path):
        lbl_path = LABELS_DIR / (img_path.stem + ".txt")
        shutil.copy2(img_path, img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    for img in train_set:
        link_pair(img, TRAIN_IMG, TRAIN_LBL)
    for img in val_set:
        link_pair(img, VAL_IMG, VAL_LBL)

    return len(train_set), len(val_set)


# ─── YAML OLUŞTURMA ───────────────────────────────────────────────────────────

def write_yaml() -> Path:
    yaml_path = DATASET_DIR / "data.yaml"
    content = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val:   images/val

nc: 3
names: ['NOMINAL', 'CAUTION', 'WARNING']
"""
    yaml_path.write_text(content)
    return yaml_path


# ─── YOLO EĞİTİMİ ─────────────────────────────────────────────────────────────

def train_model(epochs: int = 50, imgsz: int = 640, patience: int = 15) -> str:
    """
    Tam pipeline:
      1. Train/Val split
      2. data.yaml yaz
      3. YOLO eğit (erken durdurma ile)
      4. Metrikleri kaydet
      5. Model versiyonla
      6. Raporu aç

    Döndürür: model dosya yolu
    """
    summary = dataset_summary()
    if summary["total"] < 10:
        raise RuntimeError(
            f"Yetersiz veri: {summary['total']} örnek. En az 10 gerekli.\n"
            "Önce testleri çalıştırarak veri toplayın:\n"
            "  python -m pytest tests/test_ai_vision.py -v -s --no-ai"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"tusas_v{ts}"

    print(f"\n{'═'*60}")
    print(f"  TUSAŞ TestLab — YOLO Eğitimi  [{ts}]")
    print(f"  Toplam örnek : {summary['total']}")
    print(f"  NOM={summary['NOMINAL']}  CAU={summary['CAUTION']}  WARN={summary['WARNING']}")
    print(f"  Epochs={epochs}  imgsz={imgsz}  patience={patience}")
    print(f"{'═'*60}\n")

    # 1. Split
    n_train, n_val = prepare_splits()
    print(f"  Split → train={n_train}  val={n_val}")

    # 2. YAML
    yaml_path = write_yaml()

    # 3. Eğitim
    try:
        from ultralytics import YOLO
        model_path = _train_yolo_real(
            yaml_path, version_name, epochs, imgsz, patience
        )
    except ImportError:
        print("  [!] Ultralytics bulunamadı → PyTorch CNN")
        try:
            import torch
            model_path = _train_pytorch_cnn(epochs=epochs, version_name=version_name)
        except ImportError:
            raise RuntimeError(
                "\n❌ ML kütüphanesi yok.\n"
                "   pip install ultralytics          (önerilen)\n"
                "   pip install torch torchvision    (alternatif)\n"
            )

    # 4. Metrikleri kaydet
    metrics = _compute_val_metrics(model_path)
    _save_metrics(version_name, metrics, model_path, summary)

    # 5. 'latest' sembolik linki güncelle
    latest = MODEL_DIR / "latest.pt"
    shutil.copy2(model_path, latest)
    print(f"\n  ✅ Model kaydedildi : {model_path}")
    print(f"  ✅ latest.pt güncellendi")

    # 6. Eğitim raporu aç
    report_path = _generate_training_report(version_name, metrics, summary)
    _open_browser(report_path)

    return model_path


def _train_yolo_real(yaml_path, version_name, epochs, imgsz, patience) -> str:
    """YOLOv8 classification eğitimi — gerçek parametrelerle."""
    from ultralytics import YOLO

    print(f"\n  🚀 YOLO eğitimi başlıyor...")
    print(f"  Model: yolov8n-cls.pt  (en hızlı, uçuş ekranı için yeterli)")

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        lr0=0.001,
        lrf=0.01,
        patience=patience,           # Erken durdurma
        augment=True,                # YOLO dahili augmentation
        flipud=0.0,                  # Uçuş ekranı baş aşağı olmaz
        fliplr=0.3,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,                 # Küçük rotasyon
        translate=0.05,
        project=str(MODEL_DIR),
        name=version_name,
        exist_ok=True,
        verbose=True,
        plots=True,                  # Confusion matrix, PR curve kaydeder
    )

    best = MODEL_DIR / version_name / "weights" / "best.pt"
    if not best.exists():
        # Eğitim plots klasörü
        best = MODEL_DIR / version_name / "weights" / "last.pt"
    return str(best)


def _train_pytorch_cnn(epochs: int, version_name: str) -> str:
    """Saf PyTorch — Augmentation dahil."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    class ScreenDataset(Dataset):
        def __init__(self, img_dir, lbl_dir, transform=None):
            self.samples = []
            self.transform = transform
            for img_path in Path(img_dir).glob("*.png"):
                lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
                if lbl_path.exists():
                    class_id = int(lbl_path.read_text().split()[0])
                    self.samples.append((str(img_path), class_id))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ScreenDataset(TRAIN_IMG, TRAIN_LBL, train_tf)
    val_ds   = ScreenDataset(VAL_IMG,   VAL_LBL,   val_tf)

    if len(train_ds) == 0:
        raise RuntimeError("Train seti boş!")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256),nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 64),      nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 3),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CNN().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 10
    best_state = None

    print(f"  Cihaz: {device}  |  train={len(train_ds)}  val={len(val_ds)}  |  {epochs} epoch")
    _print_progress_bar(0, epochs, prefix="  Eğitim", suffix="")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += loss_fn(out, labels).item()
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        _print_progress_bar(epoch, epochs,
            prefix="  Eğitim",
            suffix=f"val_acc={val_acc*100:.1f}%  lr={opt.param_groups[0]['lr']:.5f}"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⚡ Early stopping @ epoch {epoch}  (best_val_acc={best_val_acc*100:.1f}%)")
                break

    print(f"\n  Best val accuracy: {best_val_acc*100:.1f}%")

    # En iyi ağırlıkları geri yükle
    if best_state:
        model.load_state_dict(best_state)

    version_dir = MODEL_DIR / version_name
    version_dir.mkdir(exist_ok=True)
    model_path = version_dir / "best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "classes": CLASS_NAMES,
        "val_acc": best_val_acc,
        "version": version_name,
    }, str(model_path))
    return str(model_path)


# ─── METRİKLER ────────────────────────────────────────────────────────────────

def _compute_val_metrics(model_path: str) -> dict:
    """Val seti üzerinde precision/recall/F1 hesapla."""
    preds = []
    trues = []

    for img_path in VAL_IMG.glob("*.png"):
        lbl_path = VAL_LBL / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        true_class = int(lbl_path.read_text().split()[0])
        result = predict(str(img_path), model_path)
        if "error" not in result:
            pred_class = CLASS_NAMES.index(result["class"]) if result["class"] in CLASS_NAMES else -1
            if pred_class >= 0:
                preds.append(pred_class)
                trues.append(true_class)

    if not preds:
        return {"accuracy": 0.0, "n_samples": 0, "per_class": {}}

    # Accuracy
    correct = sum(p == t for p, t in zip(preds, trues))
    accuracy = correct / len(preds)

    # Per-class precision/recall
    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        tp = sum(1 for p, t in zip(preds, trues) if p == cls_id and t == cls_id)
        fp = sum(1 for p, t in zip(preds, trues) if p == cls_id and t != cls_id)
        fn = sum(1 for p, t in zip(preds, trues) if p != cls_id and t == cls_id)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[cls_name] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   tp + fn,
        }

    return {
        "accuracy":  round(accuracy, 4),
        "n_samples": len(preds),
        "per_class": per_class,
        "confusion_matrix": _build_confusion(preds, trues),
    }


def _build_confusion(preds, trues) -> list:
    n = len(CLASS_NAMES)
    cm = [[0]*n for _ in range(n)]
    for p, t in zip(preds, trues):
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1
    return cm


def _save_metrics(version: str, metrics: dict, model_path: str, ds_summary: dict):
    record = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset": ds_summary,
        "metrics": metrics,
    }
    path = METRICS_DIR / f"{version}_metrics.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n  📊 Metrikler kaydedildi: {path}")


# ─── EĞİTİM RAPORU ────────────────────────────────────────────────────────────

def _generate_training_report(version: str, metrics: dict, ds_summary: dict) -> str:
    acc_pct = metrics.get("accuracy", 0) * 100
    n = metrics.get("n_samples", 0)
    pc = metrics.get("per_class", {})
    cm = metrics.get("confusion_matrix", [])

    # Confusion matrix HTML
    cm_html = ""
    if cm:
        header = "".join(f"<th>Pred {c}</th>" for c in CLASS_NAMES)
        cm_html = f"<table class='cm'><thead><tr><th>True \\ Pred</th>{header}</tr></thead><tbody>"
        for i, row in enumerate(cm):
            cells = ""
            for j, val in enumerate(row):
                cls = "tp" if i == j else ("fp" if val > 0 else "")
                cells += f"<td class='{cls}'>{val}</td>"
            cm_html += f"<tr><th>{CLASS_NAMES[i]}</th>{cells}</tr>"
        cm_html += "</tbody></table>"

    # Per-class tablo
    pc_rows = ""
    for cls_name, m in pc.items():
        f1_color = "#4caf50" if m["f1"] > 0.8 else ("#ff9800" if m["f1"] > 0.5 else "#f44336")
        pc_rows += f"""
        <tr>
          <td>{cls_name}</td>
          <td>{m['precision']*100:.1f}%</td>
          <td>{m['recall']*100:.1f}%</td>
          <td style="color:{f1_color}; font-weight:bold">{m['f1']*100:.1f}%</td>
          <td>{m['support']}</td>
        </tr>"""

    # Dataset dağılım chart (basit bar)
    bars = ""
    total = max(ds_summary.get("total", 1), 1)
    for cls_name, color in [("NOMINAL", "#4caf50"), ("CAUTION", "#ff9800"), ("WARNING", "#f44336")]:
        count = ds_summary.get(cls_name, 0)
        pct = count / total * 100
        bars += f"""
        <div class="bar-row">
          <span class="bar-label">{cls_name}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.0f}%; background:{color}"></div>
          </div>
          <span class="bar-count">{count}</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>TUSAŞ TestLab — Model Eğitim Raporu</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; }}
  .topbar {{ background: #111; border-bottom: 1px solid #222;
             padding: 20px 32px; display: flex; justify-content: space-between; align-items: center; }}
  .topbar h1 {{ font-size: 18px; color: #fff; }}
  .topbar .version {{ color: #666; font-size: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 28px 32px; }}
  .card {{ background: #111; border: 1px solid #222; border-radius: 10px; padding: 20px; }}
  .card h2 {{ font-size: 13px; color: #888; text-transform: uppercase;
              letter-spacing: .1em; margin-bottom: 16px; border-bottom: 1px solid #1e1e1e; padding-bottom: 10px; }}
  .big-acc {{ font-size: 64px; font-weight: bold; text-align: center; padding: 20px 0;
              color: {'#4caf50' if acc_pct >= 80 else '#ff9800' if acc_pct >= 60 else '#f44336'}; }}
  .sub-acc {{ text-align: center; color: #666; margin-top: -10px; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #1a1a1a; }}
  th {{ color: #888; font-weight: 500; }}
  td:first-child {{ text-align: left; color: #ccc; }}
  .tp {{ background: rgba(76,175,80,.15); color: #4caf50; font-weight: bold; }}
  .fp {{ background: rgba(244,67,54,.1); color: #f44336; }}
  .cm th {{ font-size: 11px; }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .bar-label {{ width: 80px; font-size: 12px; color: #aaa; }}
  .bar-track {{ flex: 1; background: #1e1e1e; height: 20px; border-radius: 4px; overflow: hidden; }}
  .bar-fill  {{ height: 100%; transition: width .5s; border-radius: 4px; }}
  .bar-count {{ width: 40px; text-align: right; font-size: 12px; color: #666; }}
  .ds-summary {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 20px; }}
  .ds-num {{ text-align: center; background: #0a0a0a; border-radius: 8px; padding: 14px; }}
  .ds-num .n {{ font-size: 28px; font-weight: bold; }}
  .ds-num .l {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .model-info {{ font-family: monospace; font-size: 12px; line-height: 1.8; color: #888; }}
  .model-info span {{ color: #ccc; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 11px;
            font-weight: bold; margin-top: 8px; }}
  .badge-good  {{ background: rgba(76,175,80,.2); color: #4caf50; }}
  .badge-warn  {{ background: rgba(255,152,0,.2); color: #ff9800; }}
  .badge-bad   {{ background: rgba(244,67,54,.2); color: #f44336; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>🛩  TUSAŞ TestLab — Model Eğitim Raporu</h1>
  <div class="version">
    Versiyon: <b style="color:#9fa8da">{version}</b> &nbsp;|&nbsp;
    {datetime.now().strftime("%d.%m.%Y %H:%M")}
  </div>
</div>

<div class="grid">
  <!-- Doğruluk -->
  <div class="card">
    <h2>Validation Doğruluğu</h2>
    <div class="big-acc">{acc_pct:.1f}%</div>
    <div class="sub-acc">{n} örnek üzerinde değerlendirme</div>
    <div style="text-align:center; margin-top:16px">
      <span class="badge {'badge-good' if acc_pct>=80 else 'badge-warn' if acc_pct>=60 else 'badge-bad'}">
        {'✅ ÜRETİME HAZIR' if acc_pct>=85 else '⚠️ İYİLEŞTİRME GEREKLİ' if acc_pct>=60 else '❌ YETERSİZ VERİ'}
      </span>
    </div>
  </div>

  <!-- Dataset -->
  <div class="card">
    <h2>Eğitim Veri Seti</h2>
    <div class="ds-summary">
      <div class="ds-num"><div class="n" style="color:#4caf50">{ds_summary.get('NOMINAL',0)}</div><div class="l">NOMINAL</div></div>
      <div class="ds-num"><div class="n" style="color:#ff9800">{ds_summary.get('CAUTION',0)}</div><div class="l">CAUTION</div></div>
      <div class="ds-num"><div class="n" style="color:#f44336">{ds_summary.get('WARNING',0)}</div><div class="l">WARNING</div></div>
    </div>
    {bars}
  </div>

  <!-- Per-class metrikler -->
  <div class="card">
    <h2>Sınıf Bazlı Metrikler</h2>
    <table>
      <thead><tr><th>Sınıf</th><th>Precision</th><th>Recall</th><th>F1</th><th>Örnek</th></tr></thead>
      <tbody>{pc_rows}</tbody>
    </table>
  </div>

  <!-- Confusion Matrix -->
  <div class="card">
    <h2>Confusion Matrix</h2>
    {cm_html if cm_html else '<p style="color:#555; text-align:center; padding:30px">Val seti boş — daha fazla veri toplayın</p>'}
  </div>
</div>

<div style="padding: 0 32px 40px">
  <div class="card">
    <h2>Model Bilgisi</h2>
    <div class="model-info">
      Versiyon  : <span>{version}</span><br>
      Sınıflar  : <span>NOMINAL / CAUTION / WARNING</span><br>
      Toplam    : <span>{ds_summary.get('total', 0)} örnek  (augmentation dahil)</span><br>
      Oluşturma : <span>{datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</span>
    </div>
  </div>
</div>
</body>
</html>"""

    report_path = METRICS_DIR / f"{version}_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  🌐 Eğitim raporu: {report_path}")
    return str(report_path)


# ─── TARAYICI AÇMA ────────────────────────────────────────────────────────────

def _open_browser(path: str):
    """Raporu/HTML'yi otomatik aç — Windows / macOS / Linux."""
    abs_path = os.path.abspath(path)
    try:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")
        print(f"  🌐 Tarayıcıda açılıyor: {abs_path}")
        return
    except Exception:
        pass

    import subprocess
    import platform
    plat = platform.system()
    try:
        if plat == "Windows":
            os.startfile(abs_path)
        elif plat == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
    except Exception as e:
        print(f"  ⚠️  Raporu manuel açın: {abs_path}")


# ─── TAHMİN ───────────────────────────────────────────────────────────────────

def predict(screenshot_path: str, model_path: Optional[str] = None) -> dict:
    """
    Eğitilmiş model ile sınıflandırma.
    Anomaly score da döndürür (confidence < 0.6 → belirsiz).
    """
    if model_path is None:
        for candidate in [
            MODEL_DIR / "latest.pt",
            *(sorted(MODEL_DIR.glob("tusas_v*/weights/best.pt"), reverse=True)[:1]),
            MODEL_DIR / "tusas_cnn.pt",
        ]:
            if Path(candidate).exists():
                model_path = str(candidate)
                break

    if not model_path or not os.path.exists(str(model_path)):
        return {"error": "Model bulunamadı. Önce 'python ml_trainer_v2.py train' çalıştırın."}

    t0 = time.time()

    # YOLO
    if "weights" in str(model_path) or "yolo" in str(model_path).lower():
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model(screenshot_path, verbose=False)
            probs = results[0].probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            anomaly = confidence < 0.6
            return {
                "class": CLASS_NAMES[class_id],
                "confidence": round(confidence, 4),
                "anomaly": anomaly,
                "ms": int((time.time() - t0) * 1000),
                "model": "yolo",
            }
        except Exception as ex:
            return {"error": f"YOLO tahmin hatası: {ex}"}

    # PyTorch
    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        ckpt = torch.load(model_path, map_location="cpu")

        class CNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                import torch.nn as nn
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256),nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(512, 64),      nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 3),
                )
            def forward(self, x):
                return self.classifier(self.features(x))

        model = CNN()
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = tf(Image.open(screenshot_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            out = model(img)
            probs = torch.softmax(out, dim=1)[0]
            class_id = int(probs.argmax())
            confidence = float(probs[class_id])

        return {
            "class": CLASS_NAMES[class_id],
            "confidence": round(confidence, 4),
            "anomaly": confidence < 0.6,
            "ms": int((time.time() - t0) * 1000),
            "model": "pytorch_cnn_v2",
        }
    except Exception as ex:
        return {"error": f"PyTorch tahmin hatası: {ex}"}


# ─── TOPLU DEĞERLENDİRME ──────────────────────────────────────────────────────

def benchmark(model_path: Optional[str] = None):
    """Tüm val seti üzerinde hız + doğruluk benchmarkı."""
    val_images = list(VAL_IMG.glob("*.png"))
    if not val_images:
        print("Val seti boş. Önce 'train' çalıştırın.")
        return

    print(f"\n  Benchmark: {len(val_images)} görüntü\n")
    times = []
    correct = 0
    anomalies = 0

    for img_path in val_images:
        lbl = VAL_LBL / (img_path.stem + ".txt")
        true_class = int(lbl.read_text().split()[0]) if lbl.exists() else -1
        result = predict(str(img_path), model_path)
        if "error" in result:
            continue
        times.append(result["ms"])
        pred_id = CLASS_NAMES.index(result["class"]) if result["class"] in CLASS_NAMES else -1
        if pred_id == true_class:
            correct += 1
        if result.get("anomaly"):
            anomalies += 1

    acc = correct / max(len(times), 1) * 100
    avg_ms = sum(times) / max(len(times), 1)
    print(f"  Accuracy  : {acc:.1f}%  ({correct}/{len(times)})")
    print(f"  Ortalama  : {avg_ms:.1f} ms/görüntü")
    print(f"  Anomaly   : {anomalies} görüntü (confidence < 0.6)")
    print(f"  Throughput: {1000/max(avg_ms,1):.0f} FPS (teorik)")


# ─── MODEL VERSİYONLARI ───────────────────────────────────────────────────────

def list_versions():
    """Eğitilmiş model sürümlerini listele."""
    metrics_files = sorted(METRICS_DIR.glob("*_metrics.json"), reverse=True)
    if not metrics_files:
        print("  Henüz eğitilmiş model yok.")
        return

    print(f"\n{'─'*65}")
    print(f"  {'Versiyon':<30} {'Accuracy':>10} {'Örnekler':>10} {'Tarih':>15}")
    print(f"{'─'*65}")
    for mf in metrics_files:
        try:
            data = json.loads(mf.read_text())
            acc  = data["metrics"].get("accuracy", 0) * 100
            n    = data["dataset"].get("total", 0)
            ts   = data.get("timestamp", "")[:16].replace("T", " ")
            ver  = data.get("version", mf.stem)
            star = " ← latest" if ver == _get_latest_version() else ""
            print(f"  {ver:<30} {acc:>9.1f}% {n:>10} {ts:>15}{star}")
        except Exception:
            pass
    print(f"{'─'*65}")


def _get_latest_version() -> str:
    latest = MODEL_DIR / "latest.pt"
    if not latest.exists():
        return ""
    mfs = sorted(METRICS_DIR.glob("*_metrics.json"), reverse=True)
    for mf in mfs:
        try:
            data = json.loads(mf.read_text())
            if data.get("model_path") and os.path.exists(data["model_path"]):
                return data.get("version", "")
        except Exception:
            pass
    return ""


# ─── DATASET SUMMARY ──────────────────────────────────────────────────────────

def dataset_summary() -> dict:
    log = _load_log()
    counts = _count_classes(log)
    return {
        "total":          len(log),
        "NOMINAL":        counts[0],
        "CAUTION":        counts[1],
        "WARNING":        counts[2],
        "ready_to_train": len(log) >= 30,
        "images_dir":     str(IMAGES_DIR),
    }


def _load_log() -> list:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _count_classes(log: list) -> Tuple[int, int, int]:
    c = [0, 0, 0]
    for e in log:
        cid = e.get("class_id", 0)
        if 0 <= cid <= 2:
            c[cid] += 1
    return tuple(c)


# ─── PROGRESS BAR ─────────────────────────────────────────────────────────────

def _print_progress_bar(current, total, prefix="", suffix="", length=30):
    filled = int(length * current / max(total, 1))
    bar = "█" * filled + "░" * (length - filled)
    pct = current / max(total, 1) * 100
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  {suffix}     ", end="", flush=True)
    if current == total:
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"

    if cmd == "summary":
        s = dataset_summary()
        print(f"\n  Veri Seti Özeti")
        print(f"  ─────────────────────────────")
        print(f"  Toplam  : {s['total']}")
        print(f"  NOMINAL : {s['NOMINAL']}")
        print(f"  CAUTION : {s['CAUTION']}")
        print(f"  WARNING : {s['WARNING']}")
        rdy = "EVET ✅" if s["ready_to_train"] else f"HAYIR (en az 30, şu an {s['total']})"
        print(f"  Hazır   : {rdy}")

    elif cmd == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        imgsz  = int(sys.argv[3]) if len(sys.argv) > 3 else 640
        try:
            path = train_model(epochs=epochs, imgsz=imgsz)
            print(f"\n  Model: {path}")
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Kullanım: python ml_trainer_v2.py predict <screenshot.png>")
            sys.exit(1)
        r = predict(sys.argv[2])
        if "error" in r:
            print(f"  HATA: {r['error']}")
        else:
            anom = "  ⚠️  ANOMALY (düşük güven)" if r.get("anomaly") else ""
            print(f"\n  Tahmin    : {r['class']}")
            print(f"  Güven     : {r['confidence']*100:.1f}%{anom}")
            print(f"  Süre      : {r['ms']}ms")
            print(f"  Model     : {r['model']}")

    elif cmd == "benchmark":
        benchmark()

    elif cmd == "versions":
        list_versions()

    else:
        print(f"Bilinmeyen komut: {cmd}")
        print("Kullanım: python ml_trainer_v2.py [summary|train|predict <dosya>|benchmark|versions]")
