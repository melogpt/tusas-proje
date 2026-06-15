"""
TUSAS TestLab — Tek Komut Test Çalıştırıcı
==========================================
Tüm testleri tek komutla çalıştır ve raporu otomatik aç.

KULLANIM:
  python run_tests.py              # Tam yerel test (Logic + Visual + ML)
  python run_tests.py --category ENGINE     # Sadece belirli kategori
  python run_tests.py --scenario ENG_001    # Tek senaryo
  python run_tests.py --v3         # Eski v3 testleri çalıştır
"""

import os
import sys
import subprocess
import platform
import webbrowser
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "test_reports")
REPORT_HTML = os.path.join(REPORT_DIR, "report.html")


def open_report():
    """Raporu tarayıcıda aç."""
    if not os.path.exists(REPORT_HTML):
        print(f"  [!] Rapor bulunamadı: {REPORT_HTML}")
        return

    opened = False
    plat = platform.system()
    if plat == "Windows":
        try:
            os.startfile(REPORT_HTML)
            opened = True
        except Exception:
            pass

    if not opened:
        try:
            opened = webbrowser.open(f"file:///{REPORT_HTML.replace(os.sep, '/')}")
        except Exception:
            pass

    if not opened:
        try:
            if plat == "Darwin":
                subprocess.Popen(["open", REPORT_HTML])
                opened = True
            elif plat != "Windows":
                subprocess.Popen(["xdg-open", REPORT_HTML])
                opened = True
        except Exception:
            pass

    url = f"file:///{REPORT_HTML.replace(os.sep, '/')}"
    if opened:
        try:
            print(f"\n  [HTML] Rapor acildi: {url}")
        except Exception:
            pass
    else:
        try:
            print(f"\n  [HTML] Rapor otomatik acilamadi, baglantiyi tiklayarak acabilirsiniz:\n  {url}")
        except Exception:
            pass


def _ensure_training_data(min_total=30):
    """Eğitim için yeterli veri yoksa sentetik üretip dengeler."""
    try:
        from ml_trainer_v3 import dataset_summary, generate_synthetic_data
    except Exception as e:
        print(f"  [ML] ml_trainer içe aktarılamadı: {e}")
        return False
    s = dataset_summary()
    need = (s["total"] < min_total) or (min(s["NOMINAL"], s["CAUTION"], s["WARNING"]) == 0)
    if need:
        print(f"  [ML] Veri yetersiz/dengesiz (total={s['total']}) → sentetik veri üretiliyor")
        generate_synthetic_data(40)
    return True


def run_real_training():
    """Testten sonra modeli GERÇEKTEN eğit (YOLO→PyTorch→sklearn sırası)."""
    print("\n" + "=" * 60)
    print("  ML — Gerçek Model Eğitimi")
    print("=" * 60)
    if not _ensure_training_data():
        return
    try:
        from ml_trainer_v3 import train_model, predict, MODEL_DIR
        mp = train_model(fast=True)
        print(f"  [ML] Egitilen model: {mp}")
        # Hızlı doğrulama: bir tahmin çalıştır
        import glob
        val = glob.glob(os.path.join(ROOT, "ml_dataset", "images", "val", "*.png"))
        if val:
            print(f"  [PREDICT] Ornek tahmin: {predict(val[0])}")
    except Exception as e:
        print(f"  [ML] Eğitim atlandı/başarısız: {type(e).__name__}: {e}")


def main():
    # Sanity check for virtual environment / dependencies
    try:
        import pytest
        import PyQt5
    except ImportError:
        print("\n" + "!" * 80)
        print("  HATA: Gerekli kütüphaneler (pytest, PyQt5) bu Python ortamında bulunamadı!")
        print("  Büyük olasılıkla VS Code'da sanal ortam (.venv) aktif ve bağımlılıklar eksik.")
        print("-" * 80)
        print("  ÇÖZÜM: Terminale 'deactivate' yazarak sanal ortamı kapatın ve ardından")
        print("  global Python ortamıyla testi çalıştırın:")
        print("  1) deactivate")
        print("  2) python run_tests.py --inject-faults --no-train")
        print("!" * 80 + "\n")
        return 1

    parser = argparse.ArgumentParser(
        description="TUSAS TestLab — Kapsamlı Local Test Çalıştırıcı"
    )
    parser.add_argument("--category", metavar="CAT",
                        help="Sadece belirli kategoriyi test et (örn. ENGINE, FUEL, COLOR)")
    parser.add_argument("--scenario", metavar="ID",
                        help="Tek senaryo çalıştır (örn. ENG_001, COLOR_002)")
    parser.add_argument("--v4", action="store_true",
                        help="Eski v4 test dosyasını çalıştır (sahte enjeksiyonlu)")
    parser.add_argument("--v3", action="store_true",
                        help="v3 test dosyasını çalıştır (eski)")
    parser.add_argument("--inject-faults", action="store_true",
                        help="Dedektör öz-testi: bilinen hatalar enjekte et, test yakalamalı")
    parser.add_argument("--no-train", action="store_true",
                        help="Testten sonra model eğitimini atla")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detaylı çıktı")
    args = parser.parse_args()

    # ── Test dosyası seçimi ────────────────────────────────────────────────────
    if args.v3:
        test_file = os.path.join(ROOT, "tests", "test_ai_vision_v3.py")
        print("  [v3] Eski test dosyası: test_ai_vision_v3.py")
    elif args.v4:
        test_file = os.path.join(ROOT, "tests", "test_ai_vision_v4.py")
        print("  [v4] Eski test dosyası: test_ai_vision_v4.py")
    else:
        test_file = os.path.join(ROOT, "tests", "test_real_vision.py")

    # ── pytest komut inşası ───────────────────────────────────────────────────
    cmd = [sys.executable, "-m", "pytest", test_file, "-s", "--tb=short",
           "-p", "no:cacheprovider"]
    cmd.append("-v" if args.verbose else "-q")
    if args.scenario:
        cmd.extend(["-k", args.scenario])
    elif args.category:
        cmd.extend(["-k", args.category])
    if args.inject_faults:
        cmd.append("--inject-faults")

    # Headless ortamda da çalışsın
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["TUSAS_NO_OPEN_REPORT"] = "1"

    print("\n" + "=" * 60)
    print("  TUSAS TestLab — Test Başlatılıyor")
    mode = "DEDEKTÖR ÖZ-TESTİ (--inject-faults)" if args.inject_faults \
           else "Gerçek Yerel Test (Piksel + Mantık + WCA)"
    print(f"  Mod: {mode}")
    if args.scenario:
        print(f"  Filtre: Senaryo = {args.scenario}")
    elif args.category:
        print(f"  Filtre: Kategori = {args.category}")
    print("=" * 60 + "\n")

    t_start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = int(time.time() - t_start)
    print(f"\n  Süre: {elapsed}s  |  Çıkış kodu: {result.returncode}")

    # ── Gerçek ML eğitimi (normal modda, varsayılan açık) ──────────────────────
    if not args.inject_faults and not args.no_train and not args.v3 and not args.v4:
        run_real_training()
        
        # Eğitim bittikten sonra Model Eğitim Raporunu da aç
        import glob
        reports = glob.glob(os.path.join(ROOT, "training_metrics", "*_report.html"))
        if reports:
            latest_report = max(reports, key=os.path.getmtime)
            try:
                webbrowser.open(f"file:///{latest_report.replace(os.sep, '/')}")
                print(f"  [HTML] Eğitim Raporu açıldı: {latest_report}")
            except Exception:
                pass

    # ── Raporu aç ────────────────────────────────────────────────────────────
    if os.path.exists(REPORT_HTML):
        time.sleep(0.5)
        open_report()

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
