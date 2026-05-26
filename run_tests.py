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
    try:
        webbrowser.open(f"file:///{REPORT_HTML.replace(os.sep, '/')}")
        opened = True
    except Exception:
        pass

    if not opened:
        try:
            plat = platform.system()
            if plat == "Windows":
                os.startfile(REPORT_HTML)
                opened = True
            elif plat == "Darwin":
                subprocess.Popen(["open", REPORT_HTML])
                opened = True
            else:
                subprocess.Popen(["xdg-open", REPORT_HTML])
                opened = True
        except Exception:
            pass

    if opened:
        print(f"\n  🌐 Rapor açıldı: {REPORT_HTML}")
    else:
        print(f"\n  Manuel açın: {REPORT_HTML}")


def main():
    parser = argparse.ArgumentParser(
        description="TUSAS TestLab — Kapsamlı Local Test Çalıştırıcı"
    )
    parser.add_argument(
        "--category",
        metavar="CAT",
        help="Sadece belirli kategoriyi test et (örn. ENGINE, FUEL, COLOR)",
    )
    parser.add_argument(
        "--scenario",
        metavar="ID",
        help="Tek senaryo çalıştır (örn. ENG_001, COLOR_002)",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="v3 test dosyasını çalıştır (eski)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Detaylı çıktı",
    )
    args = parser.parse_args()

    # ── Test dosyası seçimi ────────────────────────────────────────────────────
    if args.v3:
        test_file = os.path.join(ROOT, "tests", "test_ai_vision_v3.py")
        print("  [v3] Eski test dosyası kullanılıyor: test_ai_vision_v3.py")
    else:
        test_file = os.path.join(ROOT, "tests", "test_ai_vision_v4.py")

    # ── pytest komut inşası ───────────────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-s",           # print çıktısını göster
        "--tb=no",      # traceback kısalt (rapor var)
    ]

    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if args.scenario:
        cmd.extend(["-k", args.scenario])
    elif args.category:
        cmd.extend(["-k", args.category])

    # ── Çalıştır ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TUSAS TestLab v4 — Test Başlatılıyor")
    print("  Mod: Tam Yerel (Logic + Görsel + ML)")
    if args.scenario:
        print(f"  Filtre: Senaryo = {args.scenario}")
    elif args.category:
        print(f"  Filtre: Kategori = {args.category}")
    print("=" * 60 + "\n")

    t_start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = int(time.time() - t_start)

    print(f"\n  Süre: {elapsed}s  |  Çıkış kodu: {result.returncode}")

    # ── Raporu aç ────────────────────────────────────────────────────────────
    # pytest_sessionfinish zaten açıyor, bu backup için
    if os.path.exists(REPORT_HTML):
        time.sleep(0.5)  # Dosyanın yazılmasını bekle
        open_report()

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
