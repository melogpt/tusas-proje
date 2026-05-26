"""
TUSAŞ TestLab — YOLO Veri Toplama & Model Eğitimi
==================================================
İki aşamalı pipeline:

AŞAMA 1 — collect_training_data()
  Her test senaryosu çalışırken otomatik olarak çağrılır.
  Screenshot + gerçek etiket (scenario.severity) → YOLO formatında kaydeder.
  ~50-100 örnek toplandıktan sonra model eğitilebilir.

AŞAMA 2 — train_model()
  Toplanan veriyle küçük bir CNN sınıflandırıcı eğitir.
  (Ultralytics YOLO veya saf PyTorch, hangisi kuruluysa)
  Eğitim tamamlanınca model dosyası kaydedilir.

AŞAMA 3 — predict()
  Yeni screenshot geldiğinde <10ms'de tahmin üretir.
  Claude Vision API'ye gerek kalmaz.

Sınıflar:
  0 = NOMINAL   (tüm sistemler yeşil)
  1 = CAUTION   (sarı uyarı)
  2 = WARNING   (kırmızı kritik)

Kullanım:
  python ml_trainer.py collect   # veri topla (test runner otomatik çağırır)
  python ml_trainer.py train     # model eğit
  python ml_trainer.py predict screenshot.png  # tek görüntü tahmin et
"""

import os
import sys
import json
import time
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ─── KLASÖR YAPISI ────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "ml_dataset"
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"
MODEL_DIR   = BASE_DIR / "ml_models"
LOG_PATH    = DATASET_DIR / "collection_log.json"

for d in (IMAGES_DIR, LABELS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── SINIF HARİTASI ──────────────────────────────────────────────────────────

CLASS_MAP = {
    "NOMINAL":  0,
    "ADVISORY": 0,   # Advisory de nominal sayılır
    "CAUTION":  1,
    "WARNING":  2,
}
CLASS_NAMES = ["NOMINAL", "CAUTION", "WARNING"]

# WCA panelinin görüntüdeki normalize koordinatları (YOLO formatı: cx cy w h)
# 1600×860 baz ekran için WCA paneli: x=1200 y=680 w=395 h=175
WCA_BOX_NORMALIZED = (
    (1200 + 395/2) / 1600,   # cx
    (680  + 175/2) / 860,    # cy
    395 / 1600,               # w
    175 / 860,                # h
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
    Mevcut screenshot'ı YOLO eğitim verisine dönüştür.

    screenshot_path : test sırasında kaydedilen PNG dosyası
    scenario_id     : "ENG_001" gibi senaryo ID'si
    severity        : "WARNING" | "CAUTION" | "ADVISORY"
    """
    if not os.path.exists(screenshot_path):
        print(f"   [ML] Screenshot bulunamadı: {screenshot_path}")
        return None

    class_id = CLASS_MAP.get(severity.upper(), 0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{scenario_id}_{ts}"

    # Görüntüyü dataset/images/ altına kopyala
    dst_img = IMAGES_DIR / f"{stem}.png"
    shutil.copy2(screenshot_path, dst_img)

    # YOLO etiket dosyası: <class_id> <cx> <cy> <w> <h>
    cx, cy, bw, bh = WCA_BOX_NORMALIZED
    dst_lbl = LABELS_DIR / f"{stem}.txt"
    dst_lbl.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    entry = CollectionEntry(
        scenario_id=scenario_id,
        severity=severity,
        class_id=class_id,
        image_path=str(dst_img),
        label_path=str(dst_lbl),
        timestamp=datetime.now().isoformat(),
    )

    # Log'a ekle
    log = _load_log()
    log.append(asdict(entry))
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    counts = _count_classes(log)
    print(f"   [ML] Veri eklendi → {stem}.png  "
          f"(sınıf={CLASS_NAMES[class_id]})  "
          f"| Toplam: NOM={counts[0]} CAU={counts[1]} WARN={counts[2]}")
    return entry


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


def dataset_summary() -> dict:
    """Toplanan veri setinin özetini döndür."""
    log = _load_log()
    counts = _count_classes(log)
    return {
        "total": len(log),
        "NOMINAL": counts[0],
        "CAUTION": counts[1],
        "WARNING": counts[2],
        "ready_to_train": len(log) >= 30,
        "images_dir": str(IMAGES_DIR),
    }


# ─── MODEL EĞİTİMİ ────────────────────────────────────────────────────────────

def train_model(epochs: int = 30, imgsz: int = 640) -> str:
    """
    Toplanan veriyle model eğit.

    Önce Ultralytics YOLO'yu dener (pip install ultralytics).
    Kurulu değilse saf PyTorch CNN'e geçer.
    Kurulu değilse talimat verir.

    Döndürür: eğitilmiş model dosyasının yolu
    """
    summary = dataset_summary()
    if summary["total"] < 10:
        raise RuntimeError(
            f"Yeterli veri yok: {summary['total']} örnek var, en az 10 gerekli. "
            "Önce testleri çalıştırarak veri toplayın."
        )

    print(f"\n{'═'*55}")
    print(f"  TUSAŞ TestLab — ML Model Eğitimi")
    print(f"  Veri seti: {summary['total']} örnek")
    print(f"  NOM={summary['NOMINAL']}  CAU={summary['CAUTION']}  WARN={summary['WARNING']}")
    print(f"{'═'*55}\n")

    # ── Yol 1: Ultralytics YOLO ───────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model_path = _train_yolo(epochs=epochs, imgsz=imgsz)
        print(f"\n✅ YOLO modeli eğitildi: {model_path}")
        return model_path
    except ImportError:
        print("  [!] Ultralytics kurulu değil → PyTorch CNN deneniyor...")

    # ── Yol 2: Saf PyTorch CNN ────────────────────────────────────────────────
    try:
        import torch
        model_path = _train_pytorch_cnn(epochs=epochs)
        print(f"\n✅ PyTorch CNN modeli eğitildi: {model_path}")
        return model_path
    except ImportError:
        pass

    # ── Yol 3: Kurulum talimatı ───────────────────────────────────────────────
    msg = (
        "\n❌ ML kütüphanesi bulunamadı. Birini kurun:\n"
        "   pip install ultralytics          (YOLO — önerilen)\n"
        "   pip install torch torchvision    (PyTorch CNN)\n"
    )
    print(msg)
    raise RuntimeError(msg)


def _train_yolo(epochs: int, imgsz: int) -> str:
    """Ultralytics YOLOv8 ile classification eğitimi."""
    from ultralytics import YOLO

    # data.yaml oluştur
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: images
val: images

nc: 3
names: ['NOMINAL', 'CAUTION', 'WARNING']
"""
    yaml_path.write_text(yaml_content)

    # En küçük YOLOv8 classification modeli
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        project=str(MODEL_DIR),
        name="tusas_testlab",
        exist_ok=True,
        verbose=False,
    )

    best = MODEL_DIR / "tusas_testlab" / "weights" / "best.pt"
    return str(best)


def _train_pytorch_cnn(epochs: int) -> str:
    """Saf PyTorch ile küçük CNN sınıflandırıcı."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    class ScreenDataset(Dataset):
        def __init__(self, transform=None):
            self.samples = []
            self.transform = transform
            log = _load_log()
            for e in log:
                img_p = e.get("image_path", "")
                if os.path.exists(img_p):
                    self.samples.append((img_p, e["class_id"]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB").resize((224, 224))
            if self.transform:
                img = self.transform(img)
            return img, label

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = ScreenDataset(transform=transform)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

    # Basit CNN: 3 conv + 2 fc
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 3),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SimpleCNN().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Cihaz: {device}  |  {len(dataset)} örnek  |  {epochs} epoch")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
        if epoch % 5 == 0 or epoch == epochs:
            acc = correct / max(total, 1) * 100
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(loader):.4f}  acc={acc:.1f}%")

    model_path = MODEL_DIR / "tusas_cnn.pt"
    torch.save({"model_state": model.state_dict(), "classes": CLASS_NAMES}, str(model_path))
    return str(model_path)


# ─── TAHMİN ───────────────────────────────────────────────────────────────────

def predict(screenshot_path: str, model_path: Optional[str] = None) -> dict:
    """
    Eğitilmiş modelle tek screenshot'ı sınıflandır.

    Döndürür: {"class": "WARNING", "confidence": 0.97, "ms": 8}
    """
    if model_path is None:
        # Otomatik bul
        for candidate in [
            MODEL_DIR / "tusas_testlab" / "weights" / "best.pt",
            MODEL_DIR / "tusas_cnn.pt",
        ]:
            if candidate.exists():
                model_path = str(candidate)
                break

    if not model_path or not os.path.exists(str(model_path)):
        return {"error": "Model bulunamadı. Önce 'python ml_trainer.py train' çalıştırın."}

    t0 = time.time()

    # YOLO tahmini
    if "tusas_testlab" in str(model_path) or "yolo" in str(model_path).lower():
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model(screenshot_path, verbose=False)
            probs = results[0].probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            return {
                "class": CLASS_NAMES[class_id],
                "confidence": round(confidence, 4),
                "ms": int((time.time() - t0) * 1000),
                "model": "yolo",
            }
        except Exception as ex:
            return {"error": f"YOLO tahmin hatası: {ex}"}

    # PyTorch tahmini
    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        checkpoint = torch.load(model_path, map_location="cpu")

        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                import torch.nn as nn
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 3),
                )
            def forward(self, x):
                return self.classifier(self.features(x))

        model = SimpleCNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = transform(Image.open(screenshot_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            out = model(img)
            probs = torch.softmax(out, dim=1)[0]
            class_id = int(probs.argmax())
            confidence = float(probs[class_id])

        return {
            "class": CLASS_NAMES[class_id],
            "confidence": round(confidence, 4),
            "ms": int((time.time() - t0) * 1000),
            "model": "pytorch_cnn",
        }
    except Exception as ex:
        return {"error": f"PyTorch tahmin hatası: {ex}"}


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"

    if cmd == "summary":
        s = dataset_summary()
        print(f"\nVeri Seti Özeti")
        print(f"  Toplam  : {s['total']}")
        print(f"  NOMINAL : {s['NOMINAL']}")
        print(f"  CAUTION : {s['CAUTION']}")
        print(f"  WARNING : {s['WARNING']}")
        print(f"  Eğitime hazır: {'EVET' if s['ready_to_train'] else f'HAYIR (en az 30 örnek, şu an {s[chr(116)+(chr(111)+chr(116)+(chr(97)+chr(108)))]})' }")
        print(f"  Klasör  : {s['images_dir']}")

    elif cmd == "train":
        ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        try:
            path = train_model(epochs=ep)
            print(f"\nModel kaydedildi: {path}")
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Kullanım: python ml_trainer.py predict <screenshot.png>")
            sys.exit(1)
        result = predict(sys.argv[2])
        if "error" in result:
            print(f"HATA: {result['error']}")
        else:
            print(f"\nTahmin : {result['class']}")
            print(f"Güven  : {result['confidence']*100:.1f}%")
            print(f"Süre   : {result['ms']}ms")
            print(f"Model  : {result['model']}")

    else:
        print(f"Bilinmeyen komut: {cmd}")
        print("Kullanım: python ml_trainer.py [summary|train|predict <dosya>]")
