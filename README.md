# TUSAS TestLab — Flight Systems Display
## v5 — Dürüst Render Testi, Hata Enjeksiyonu + Gerçek Model Eğitimi

Bir helikopterin/uçağın uçuş, motor, çevre kontrol ve elektrik sistemlerini
görselleştiren PyQt5 arayüzü. Sesli asistan + AI Vision + **gerçekten eğitilen
ML modeli** entegre. Test katmanı artık kendi enjekte ettiği hatayı değil,
**arayüzün gerçek render doğruluğunu** ölçer.

> Detaylı sürüm gelişimleri için bkz. `DEGISIKLIKLER.md`.

---

## Kurulum ve Bağımlılıklar

```bash
pip install -r requirements.txt
```

Sanal ortam veya sistem düzeyinde kurulması gereken tüm bileşenler aşağıda belirtilmiştir:

### 1. Sistem Bağımlılıkları (Kullanılacak özelliklere göre):

| Özellik | Linux | macOS | Windows |
|---|---|---|---|
| **Ses (PyAudio)** | `sudo apt install portaudio19-dev` | `brew install portaudio` | Hazır `.whl` paketleri kurulur.* |
| **OCR (pytesseract)** | `sudo apt install tesseract-ocr` | `brew install tesseract` | [UB-Mannheim Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) `.exe` indirip kurun ve sistem PATH'ine ekleyin. |

> **\* ÖNEMLİ (VS Code & .venv Uyarı):**  
> VS Code'un otomatik olarak aktif ettiği sanal ortamlarda Python sürümü **3.14 (Preview/Dev)** ise, Windows için `PyAudio` derleme/kütüphane bağımlılığı hatası verebilir.  
> Bu durumda terminale **`deactivate`** yazarak sanal ortamı kapatıp testleri kütüphanelerin yüklü olduğu global Python ortamında çalıştırabilirsiniz.

### 2. ML & Raporlama Modülleri:
- **`scikit-learn`** (Zorunlu): RandomForestClassifier kullanarak hafif model eğitimini sağlar. `ultralytics` veya `torch` kütüphaneleri kurulmadığında otomatik fallback olarak devreye girer.
- **`ultralytics` & `torch`** (Opsiyonel): Gelişmiş YOLOv8 tespiti ve model eğitimi gerçekleştirmek isterseniz kurabilirsiniz.
- **`pytesseract`**: Ekranda çizilen sayısal değerlerin gerçek piksel analizini (OCR) gerçekleştirmek için kullanılır.
- **`reportlab`**: PDF raporlama modülüdür.

---

## Uygulamayı Çalıştırma

```bash
python ekran.py
```

---

## Test Sistemi — Dürüst Mimari

Eski testler senaryoyu uyguladıktan sonra widget'ları **bilerek bozup** o hatayı
"yakalıyordu" — yani UI doğruluğunu hiç ölçmüyordu. v3'te test, birbirinden
**bağımsız iki doğruluk kaynağı** kullanır; kendini doğrulayamaz:

```
Ekran görüntüsü (gerçek render)
  ├── Piksel sadakati    — bar rengi/doluluğu ↔ widget'ın amaçladığı değer
  │                        + OCR ile okunan sayı (tesseract, kanıt)
  ├── Bağımsız mantık    — widget.get_state(value) ↔ param_config eşikleri
  │                        + WCA mesajları & Master Caution ↔ senaryo beklentisi
  └── Claude Vision       — doğal dil görsel doğrulama (opsiyonel, çevrimiçi)
```

### Tek komutla tam test ve raporlama

```bash
# Testleri çalıştırır, ekran görüntülerini yakalar, HTML raporu üretir ve otomatik olarak tarayıcıda açar:
python run_tests.py
```

`run_tests.py` varsayılan olarak **`tests/test_real_vision.py`** dosyasını çalıştırır, ardından modeli gerçekten eğitir ve ekran görüntüleriyle birlikte oluşan interaktif HTML raporunu tarayıcıda otomatik olarak açar. Rapor dosyası `test_reports/report.html` konumundadır.

> **ÖNEMLİ (Ekran Görüntüsü ve Dosya Temizliği):**
> Testler sırasında alınan tüm ekran görüntüleri (screenshot), işaretli hata kutuları (annotated) ve kırpılmış parametre görselleri (crops) **HTML raporunun içerisine base64 formatında doğrudan gömülür (embedded)**. 
> Dosya boyutunun orantısız artmasını önlemek amacıyla, testler bittikten hemen sonra diskteki geçici screenshot (`.png`) dosyaları **otomatik olarak temizlenir**. Böylece raporunuz tüm görselleriyle tek bir dosya olarak eksiksiz açılırken, diskiniz şişmez.

> **Windows / PowerShell:** komutu mutlaka `python` ile başlatın. Örneğin: `python run_tests.py --no-train`.

| Komut | Açıklama |
|---|---|
| `python run_tests.py` | Dürüst test (piksel + mantık + WCA) + gerçek model eğitimi + interaktif HTML raporunu açar |
| `python run_tests.py --inject-faults --no-train` | **Dedektör Öz-Testi:** 34 senaryoya dinamik olarak çeşitlendirilmiş hatalar enjekte eder ve raporu açar. (33 hata yakalanır, `VIS_001` senaryosundaki `'CALIBRATION REQUIRED'` gizli anomalisi yakalanmayarak raporda kaçırıldığı belirtilir) |
| `python run_tests.py --no-train` | Sadece dürüst testleri çalıştırır ve raporu açar (model eğitimi adımı atlanır) |
| `python run_tests.py --scenario ENG_002` | Sadece belirli bir senaryoyu (örn: ENG_002) çalıştırır ve raporunu açar |
| `python run_tests.py --category ENGINE` | Sadece belirli bir kategoriye (örn: ENGINE) ait senaryoları çalıştırır |
| `python run_tests.py --v3` / `--v4` | Karşılaştırma amacıyla eski (sahte enjeksiyonlu) test dosyalarını çalıştırır |
| `python run_tests.py -v` | Testlerin konsol çıktısını detaylandırır |

### Sadece test — modeli / YOLO çalıştırmadan

ML modelini (YOLO/PyTorch/sklearn) **hiç çalıştırmadan** yalnızca testi koşup raporu açmak için iki yol var:

```bash
# 1) run_tests.py üzerinden — eğitim adımı atlanır (rapor otomatik açılır)
python run_tests.py --no-train

# 2) Doğrudan pytest üzerinden çalıştırma (rapor otomatik açılır)
python -m pytest tests/test_real_vision.py -v -s
```

### Doğrulanmış sonuçlar
- **Normal mod:** 34/34 senaryo başarıyla geçti (358 hard-check, 0 beklenmeyen hata).
- **`--inject-faults` modu:** 34/34 senaryo başarıyla çalıştı (33 enjekte hata yakalandı, 1 missed anomali başarıyla raporlandı).

---

## ML Model Eğitimi (v3)

Backend seçimi ortama göre otomatiktir: **ultralytics → PyTorch → scikit-learn**.
Hangisi kuruluysa o devreye girer; en kötü ihtimalde scikit-learn ile renk-
öznitelik tabanlı bir RandomForest **gerçekten** eğitilir.

### Pipeline

```
Testler çalışır
  → Her senaryonun gerçek ekran görüntüsü ml_dataset/ içine toplanır
  → Veri azsa sentetik üretilip sınıflar dengelenir
  → Stratified train/val ayrımı uygulanır
  → Model eğitilir (gerçek accuracy/precision/recall/F1/confusion)
  → latest.pt (YOLO/torch) veya latest.pkl (sklearn) kaydedilir
  → Rapor otomatik açılır
```

### `ml_trainer_v3.py` komutları

```bash
python ml_trainer_v3.py summary                    # Veri durumu
python ml_trainer_v3.py generate 40                # Sentetik veri üret
python ml_trainer_v3.py train                      # Model eğit (varsayılan 50 epoch)
python ml_trainer_v3.py train 100                  # Epoch özelleştir
python ml_trainer_v3.py predict <ekran.png>        # Tek görüntü tahmini
python ml_trainer_v3.py benchmark                  # Val seti hız + doğruluk
python ml_trainer_v3.py versions                   # Eğitilmiş sürümleri listele
python ml_trainer_v3.py colab                      # Colab eğitim notebook'u üret
```

### Çıktılar

```
ml_models/
  ├── latest.pt   veya   latest.pkl     ← Her zaman güncel model (backend'e göre)
  └── tusas_v<tarih>/ ...               ← Tarih damgalı sürümler

training_metrics/
  ├── *_metrics.json                    ← Precision/Recall/F1/Confusion
  └── *_report.html                     ← Otomatik açılan rapor
```

---

## Anahtar Özellikler (v3)

| Özellik | Açıklama |
|---|---|
| **Dürüst render testi** | Test kendi hatasını değil, gerçek pikselleri doğrular |
| **OCR kanıtı** | tesseract ile bar üzerindeki sayı/birim okunur |
| **Bağımsız mantık kaynağı** | `param_config` eşikleri widget'tan ayrı tutulur |
| **Dedektör öz-testi** | `--inject-faults` ile testin gerçekten yakaladığı kanıtlanır |
| **Çok-backend eğitim** | ultralytics → PyTorch → scikit-learn otomatik düşüş |
| **Garantili eğitim** | Ağır kütüphane olmadan da model gerçekten eğitilir |
| **Gerçek metrikler** | Stratified split, accuracy/precision/recall/F1/confusion |
| **Anomaly Detection** | confidence < 0.6 ise ⚠️ anomaly işaretlenir |
| **Otomatik rapor** | Test ve eğitim sonrası HTML rapor otomatik açılır |
| **Cross-platform** | Windows / macOS / Linux + başsız (offscreen) destek |

---

## Test Senaryoları

| Kategori | Açıklama |
|---|---|
| ENGINE | Motor torku, sıcaklık, yağ basıncı |
| FUEL | Yakıt seviyesi, dengesizlik, basınç |
| ELECTRICAL | Jeneratör voltajı, AC frekansı, DC bus |
| ENVIRONMENTAL | Kabin basıncı, duman, bleed air, diferansiyel |
| HYDRAULIC | Hidrolik sistem basıncı (A/B) |
| ROTOR | Vibrasyon, devir (Nr) düşüklüğü |
| CASCADE | Çoklu eş zamanlı arıza |
| NOMINAL | Baz durum — tüm sistemler yeşil |

> Senaryo seti `tests/fault_scenarios.py` + `tests/test_ai_vision_v4.py`
> (tek kaynak) üzerinden `ALL_SCENARIOS` olarak toplanır ve
> `test_real_vision.py` içinde parametrize edilir.

---

## Üst Seviye Geliştirme Fikirleri

1. **Temporal Detection** — Ardışık frameler arasındaki değişim hızına bakarak "hızlı bozulma" alarmı
2. **Multi-label** — Tek ekranda birden fazla sistem hatası sınıflandırması
3. **Transfer Learning** — TUSAŞ-spesifik pre-trained backbone (sentetik veri ile)
4. **Active Learning** — Modelin en emin olmadığı örnekleri test öncelik listesine koy
5. **ONNX Export** — `best.pt → best.onnx` → gömülü sistemde çalışma
6. **Regression Branch** — Sınıflandırma değil, anlık sensör değeri tahmini
7. **Trend Analysis** — 10 ardışık tahmin üzerinde sliding window, erken uyarı sistemi