"""
TUSAS TestLab — Pytest Global Configuration
"""

import os
import platform
import subprocess
import webbrowser


def pytest_addoption(parser):
    pass


def pytest_sessionfinish(session, exitstatus):
    """
    Tüm testler bittikten sonra raporu otomatik aç.
    Bu hook sadece test_ai_vision_v4.py içinde kendi hook'u yoksa devreye girer
    (v4 zaten kendi hook'unu tanımlıyor; bu v3 ve diğerleri için fallback).
    """
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
        print(f"\n  🌐 Rapor açıldı: {html_path}")
    else:
        print(f"\n  Manuel açın: {html_path}")
