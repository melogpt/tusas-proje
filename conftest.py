"""
TUSAS TestLab — Pytest Global Configuration
"""

import os
import platform
import subprocess
import webbrowser


import pytest

_global_app = None

@pytest.fixture(scope="session", autouse=True)
def load_offscreen_fonts():
    global _global_app
    # Load fonts in QApplication if running in offscreen mode on Windows to enable text rendering
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and platform.system() == "Windows":
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QFontDatabase
            # Initialize or get QApplication instance and keep reference to prevent garbage collection
            _global_app = QApplication.instance()
            if _global_app is None:
                import sys
                _global_app = QApplication(sys.argv)
            db = QFontDatabase()
            if not db.families():
                for font_file in ["consola.ttf", "consolab.ttf", "arial.ttf", "arialbd.ttf"]:
                    path = os.path.join("C:/Windows/Fonts", font_file)
                    if os.path.exists(path):
                        db.addApplicationFont(path)
        except Exception:
            pass


def pytest_addoption(parser):
    parser.addoption(
        "--inject-faults", action="store_true", default=False,
        help="Dedektör öz-testi: UI'a bilinen bir render hatası enjekte et; "
             "dürüst test bunu YAKALAMALI (en az bir hard-check FAIL vermeli).",
    )
    parser.addoption(
        "--no-ai", action="store_true", default=False,
        help="AI analizlerini devre disi birak",
    )


def pytest_sessionstart(session):
    import shutil
    report_dir = os.path.join(os.path.dirname(__file__), "test_reports")
    screenshots_dir = os.path.join(report_dir, "screenshots")
    if os.path.exists(screenshots_dir):
        shutil.rmtree(screenshots_dir)
    os.makedirs(screenshots_dir, exist_ok=True)

def pytest_sessionfinish(session, exitstatus):
    """
    Tüm testler bittikten sonra raporu otomatik aç.
    Bu hook sadece test_ai_vision_v4.py içinde kendi hook'u yoksa devreye girer
    (v4 zaten kendi hook'unu tanımlıyor; bu v3 ve diğerleri için fallback).
    """
    if os.environ.get("TUSAS_NO_OPEN_REPORT") == "1":
        return
    report_dir = os.path.join(os.path.dirname(__file__), "test_reports")
    html_path = os.path.abspath(os.path.join(report_dir, "report.html"))

    if not os.path.exists(html_path):
        return

    # v4 zaten kendi açma logic'ini çalıştırıyor — duplicate açma yapma
    # Sadece doğrudan v3/sistem testi çalıştırıldıysa buradan aç
    collected = getattr(session, "_collect_items_count", None)
    if collected is not None:
        return

    _open_html(html_path)


def _open_html(html_path: str):
    opened = False
    try:
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
        opened = True
    except Exception:
        pass

    if not opened:
        try:
            plat = platform.system()
            if plat == "Windows":
                os.startfile(html_path)
                opened = True
            elif plat == "Darwin":
                subprocess.Popen(["open", html_path])
                opened = True
            else:
                subprocess.Popen(["xdg-open", html_path])
                opened = True
        except Exception:
            pass

    if opened:
        try:
            print(f"\n  [HTML] Rapor acildi: {html_path}")
        except Exception:
            pass
    else:
        try:
            print(f"\n  Manuel acin: {html_path}")
        except Exception:
            pass
