# TUSAS TestLab — Flight Systems Display  
## v2 — Gerçek YOLO Eğitimi + Otomatik Raporlama

Bir helikopterin/uçağın uçuş, motor, çevre kontrol ve elektrik sistemlerini
görselleştiren PyQt5 arayüzü. Sesli asistan + AI Vision + **gerçek YOLO modeli** entegre.

---

## Kurulum

```bash
pip install -r requirements.txt
```

> PyAudio sorunu çıkarsa:
> - Linux: `sudo apt install portaudio19-dev`
> - macOS: `brew install portaudio`

---

## Uygulamayı Çalıştırma

```bash
python ekran.py
```

---

## Test Sistemi — Üç Katman

```
Screenshot
  ├── OpenCV  (~5ms)    — offline renk analizi
  ├── ML v2   (~10ms)   — YOLOv8 veya PyTorch CNN
  └── Claude Vision     — doğal dil görsel doğrulama
```

### Hızlı Test (logic + screenshot, ~15 sn)
```bash
python -m pytest tests/test_ai_vision_v2.py -v -s --no-ai
```

### Tam Test (Claude Vision dahil, ~90 sn)
```bash
python -m pytest tests/test_ai_vision_v2.py -v -s
```

> Testler tamamlanınca **rapor otomatik olarak tarayıcıda açılır.**  
> Manuel: `test_reports/report.html`

---

## ML Model Eğitimi v2

### Pipeline

```
Testler çalışır
  → Her senaryo 4 örnek üretir (orijinal + 3 augmented)
  → ml_dataset/images/all/ klasörüne kaydedilir
  → Eğitimde train/val split uygulanır (%80 / %20)
  → YOLO eğitilir (early stopping, augmentation)
  → Eğitim sonunda rapor otomatik açılır
```

 # tusas-testlab test
 python run_tests.py
 # accuracy
 python ml_trainer_v3.py benchmark

```bash
# Data status
python ml_trainer_v2.py summary

# Model eğit (önerilen)
python ml_trainer_v2.py train

# Epoch ve imgsz özelleştir
python ml_trainer_v2.py train 100 416

# Tek görüntü tahmin et
python ml_trainer_v2.py predict test_reports/screenshots/ENG_001.png

# Val seti üzerinde benchmark (hız + doğruluk)
python ml_trainer_v2.py benchmark

# Tüm eğitilmiş versiyonları listele
python ml_trainer_v2.py versions
```

### Eğitim Çıktıları

```
ml_models/
  ├── latest.pt                     ← Her zaman güncel model
  ├── tusas_v20260430_143022/
  │   └── weights/
  │       ├── best.pt               ← En iyi epoch
  │       └── last.pt
  └── ...

training_metrics/
  ├── tusas_v20260430_143022_metrics.json   ← Precision/Recall/F1
  └── tusas_v20260430_143022_report.html   ← Otomatik açılan rapor
```

---

## Yeni Özellikler (v2)

| Özellik | Açıklama |
|---|---|
| **Augmentation** | Her screenshot → 4 örnek (brightness ±, flip) |
| **Train/Val split** | %80/%20, seed sabitli, tekrarlanabilir |
| **Early Stopping** | Val accuracy iyileşmezse eğitim durur |
| **Model Versioning** | Tarih damgalı sürümler, `latest.pt` linki |
| **Anomaly Detection** | confidence < 0.6 ise ⚠️ anomaly işaretlenir |
| **Eğitim Raporu** | Accuracy, F1, Confusion Matrix → otomatik açılan HTML |
| **Benchmark** | Val seti tüm tahminleri, FPS ölçümü |
| **Cross-platform rapor** | Windows / macOS / Linux otomatik açılır |

---

## Test Senaryoları

| Kategori | Senaryo | Açıklama |
|---|---|---|
| ENGINE | 5 | Motor torku, sıcaklık, yağ basıncı |
| FUEL | 3 | Yakıt seviyesi, dengesizlik, basınç |
| ELECTRICAL | 2 | Jeneratör voltajı, AC frekansı |
| ENVIRONMENTAL | 3 | Kabin basıncı, duman, bleed air |
| HYDRAULIC | 1 | Hidrolik sistem basıncı |
| ROTOR | 2 | Vibrasyon, devir düşüklüğü |
| CASCADE | 2 | Çoklu eş zamanlı arıza |
| NOMINAL | 1 | Baz durum — tüm sistemler yeşil |

---

## Üst Seviye Geliştirme Fikirleri

1. **Temporal Detection** — Ardışık frameler arasındaki değişim hızına bakarak "hızlı bozulma" alarmı
2. **Multi-label** — Tek ekranda birden fazla sistem hatası sınıflandırması
3. **Transfer Learning** — TUSAŞ-spesifik pre-trained backbone (sentetik veri üretimi ile)
4. **Active Learning** — Modelin en emin olmadığı örnekleri test öncelik listesine koy
5. **ONNX Export** — `best.pt → best.onnx` → Gömülü sistemde çalışma
6. **Regression Branch** — Sınıflandırma değil, anlık sensör değeri tahmini
7. **Trend Analysis** — 10 ardışık tahmin üzerinde sliding window, erken uyarı sistemi
