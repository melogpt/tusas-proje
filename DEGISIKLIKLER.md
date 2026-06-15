# TUSAS TestLab — Dürüst Test & Gerçek Eğitim Güncellemesi

> UI dosyalarına (ekran.py, widgets.py, models.py, dialogs.py, utils.py …)
> **hiç dokunulmadı** — hepsi yüklenen orijinalle byte-byte aynı.

## 1. Sorun: Test "yapıyormuş gibi yapıyordu"
Eski `tests/test_ai_vision_v4.py`, senaryoyu uyguladıktan sonra
`inject_random_visual_bug()` ile widget'ları **bilerek bozuyordu**; sonra
"hata buldum" diyordu. Yani test kendi enjekte ettiği hatayı yakalıyor, UI'ın
doğruluğunu hiç ölçmüyordu. Ayrıca:
- ML modeli akış içinde **hiç eğitilmiyordu** (sadece `predict` çağrılıyor, model yoksa "Model yok" dönüyordu).
- Sayısal kontroller render edilen pikselleri değil, Qt veri modelini (`lbl_val.text()`) okuyordu → gerçek OCR yoktu.

## 2. Yeni dürüst test mimarisi
İki **bağımsız** doğruluk kaynağı kullanılır; test kendini doğrulayamaz:

1. **Render sadakati (pikseller):** bar piksel rengi ↔ widget'ın amaçladığı renk;
   piksel doluluğu ↔ değerden hesaplanan doluluk; OCR ile okunan sayı (kanıt).
2. **Mantık:** `widget.get_state(value)` ↔ **bağımsız** `param_config` eşikleri;
   WCA mesajları + Master Caution ↔ senaryo beklentisi.

### Yeni / değişen dosyalar
- `tests/ocr_utils.py` — **OCR** modülü (tesseract). Beyaz-koyu Consolas için
  ayarlı ön-işleme; sayı/birim okuma. Bağımlılık yoksa zarifçe atlar.
- `tests/harness.py` — **deterministik düzenek**. `deterministic_apply()`
  senaryoyu uygulamanın **gerçek** `_tick_sim → FaultGate → WcaStore` hattıyla,
  rastgelelik olmadan uygular. `_LockedDict`, türetilmiş hesapların enjekte
  edilen arıza değerini ezmesini önler → WCA gerçek arızadan dolar.
  Kalıcı "DEMO" WCA girdileri `real_wca_entries()` ile ayıklanır.
- `tests/real_checks.py` — **dürüst kontroller**: `visibility`, `model_value`
  (değer **ve birim**; INVALID parametrede geçersizlik göstergesi), `ocr_render`
  (bilgilendirici), `bar_color_render`, `bar_fill_render`, `state_logic`,
  `master_caution`, `wca_present`, `wca_no_spurious`.
- `tests/test_real_vision.py` — **yeni pytest**. Tüm senaryolar parametrize;
  her senaryoda piksel + mantık + WCA kontrolleri; gerçek ekran görüntüleri ML
  veri setine toplanır; kendi içinde HTML+JSON rapor üretir.

### Dedektör öz-testi (testin boş olmadığının kanıtı)
```
python run_tests.py --inject-faults
```
Her senaryoya, belirli bir kontrole eşlenen **bilinen** bir render hatası
(yanlış sayı/birim/doluluk, INVALID'de sahte değer) enjekte edilir ve dürüst
testin bunu **yakalaması** beklenir. Doğrulandı:
- Normal mod: **34/34 senaryo geçti** (doğru UI, 358 hard-check, 0 hata).
- `--inject-faults`: **34/34 enjekte hata yakalandı**.

## 3. Model artık GERÇEKTEN eğitiliyor
`ml_trainer_v3.py`:
- Yeni **scikit-learn RandomForest** backend'i (`_train_sklearn`, `_extract_features`).
  Eğitim sırası: **ultralytics → PyTorch → scikit-learn**. torch/ultralytics
  olmayan ortamda bile model gerçekten eğitilir.
- Öznitelikler renk-tabanlı: genel + WCA bölgesi kırmızı/sarı/yeşil oranları,
  4×4 ızgara ortalama RGB. Stratified train/val ayrımı, gerçek
  accuracy/precision/recall/F1/confusion (`_compute_metrics` her backend için).
- `predict()` artık `.pkl` (sklearn) modellerini de yükler.
- `run_tests.py` testten sonra `train_model()` çağırır; veri azsa sentetik
  üretip dengeler. (Doğrulandı: model eğitildi, `latest.pkl` kaydedildi, örnek
  tahmin `WARNING @ %97.5` döndü.)

> Not: Renk sınıf sinyali çok güçlü olduğundan doğrulama doğruluğu yüksektir;
> bu kolay 3-sınıflı renk görevinde beklenen bir durumdur.

## 4. Kullanım
```
python run_tests.py                 # dürüst test + gerçek eğitim
python run_tests.py --inject-faults # dedektör öz-testi (eğitim atlanır)
python run_tests.py --no-train      # sadece test
python run_tests.py --scenario ENG_002
python run_tests.py --v4            # eski (sahte) test, karşılaştırma için
python ml_trainer_v3.py predict <ekran.png>
```
Başsız (sunucu) ortamda `QT_QPA_PLATFORM=offscreen` otomatik ayarlanır.
