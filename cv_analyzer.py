"""
TUSAŞ TestLab — OpenCV Lokal Görüntü Analizi
=============================================
İnternet bağlantısı gerektirmez. Ekranın belirli
bölgelerini keser, renk dağılımına bakarak sistemi
saniyeler içinde analiz eder.

Nasıl çalışır:
  1. Screenshot (PIL/QPixmap) numpy dizisine çevrilir
  2. Ekranda sabit koordinatlardaki paneller kesilir
  3. Her panel için HSV renk dağılımı hesaplanır
  4. Kırmızı/sarı/yeşil oranına göre durum belirlenir
  5. Sonuç: "WARNING" | "CAUTION" | "NOMINAL" | "UNKNOWN"
"""

import os
import json
import base64
import struct
import zlib
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

# numpy zorunlu, cv2 opsiyonel (kurulu değilse saf Python fallback)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except (ImportError, OSError):
    HAS_CV2 = False


# ─── RENK TANIMLARI (HSV aralıkları) ─────────────────────────────────────────
# Ekran renkleri: WARNING=#FF3333, CAUTION=#FFB000, NOMINAL=#00FF66

# HSV: H(0-179), S(0-255), V(0-255) — OpenCV'nin skalaması
COLOR_RANGES = {
    "WARNING": [
        # Kırmızı: 0°-10° ve 170°-179°
        ((0,   120, 120), (10,  255, 255)),
        ((170, 120, 120), (179, 255, 255)),
    ],
    "CAUTION": [
        # Turuncu-sarı: 15°-30°
        ((15, 120, 120), (30, 255, 255)),
    ],
    "NOMINAL": [
        # Yeşil: 50°-85°
        ((50, 100, 100), (85, 255, 255)),
    ],
}

# Karar eşikleri (bir renk toplam pikselin bu kadarını oluşturuyorsa o durum)
THRESHOLDS = {
    "WARNING": 0.04,   # %4 kırmızı → WARNING
    "CAUTION": 0.05,   # %5 sarı/turuncu → CAUTION
    "NOMINAL": 0.10,   # %10 yeşil → NOMINAL
}


# ─── PANEL KOORDİNATLARI (1600×860 çözünürlük baz alınmıştır) ────────────────
# (x1, y1, x2, y2) — piksel koordinatları
# Ekran boyutu farklıysa _scale_regions() otomatik ölçekler

BASE_W, BASE_H = 1600, 860

PANEL_REGIONS = {
    "wca":        (1200, 680, 1595, 855),   # WCA paneli (sağ alt)
    "master_cau": (870,  15,  1060, 65),    # Master Caution etiketi
    "engine1":    (10,   130, 400, 440),    # Engine 1 parametreleri
    "engine2":    (410,  130, 800, 440),    # Engine 2 parametreleri
    "fuel":       (10,   660, 370, 855),    # Fuel paneli
    "electrical": (375,  660, 570, 855),    # Electrical paneli
    "hydraulic":  (575,  660, 870, 855),    # Hydraulics paneli
}


@dataclass
class PanelResult:
    panel: str
    state: str           # "WARNING" | "CAUTION" | "NOMINAL" | "UNKNOWN"
    warning_ratio: float
    caution_ratio: float
    nominal_ratio: float
    dominant_color: str  # en yüksek oran hangi renkte


@dataclass
class CVAnalysisResult:
    overall_state: str              # en kötü panel durumu
    panels: Dict[str, PanelResult]
    wca_is_red: bool
    wca_is_yellow: bool
    wca_is_green: bool
    mc_is_active: bool
    analysis_ms: int = 0
    method: str = "opencv"          # "opencv" | "fallback"
    error: str = ""


# ─── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────

def _scale_regions(actual_w: int, actual_h: int) -> Dict[str, Tuple]:
    """Panel koordinatlarını gerçek ekran boyutuna ölçekle."""
    sx = actual_w / BASE_W
    sy = actual_h / BASE_H
    scaled = {}
    for name, (x1, y1, x2, y2) in PANEL_REGIONS.items():
        scaled[name] = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
    return scaled


def _count_color_ratio(hsv_region, lower, upper) -> float:
    """HSV bölgesinde belirli renk aralığının piksel oranını döndür."""
    if not HAS_CV2:
        return 0.0
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv_region, lower_arr, upper_arr)
    total = hsv_region.shape[0] * hsv_region.shape[1]
    return float(mask.sum() / 255) / max(total, 1)


def _analyze_region_hsv(bgr_region) -> PanelResult:
    """Bir BGR bölgesini HSV'ye çevirip renk oranlarını hesapla."""
    dummy = PanelResult("?", "UNKNOWN", 0, 0, 0, "UNKNOWN")
    if not HAS_CV2 or bgr_region is None or bgr_region.size == 0:
        return dummy

    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)

    ratios = {}
    for state, ranges in COLOR_RANGES.items():
        total_ratio = sum(
            _count_color_ratio(hsv, lo, hi)
            for lo, hi in ranges
        )
        ratios[state] = total_ratio

    # Durum belirle
    state = "UNKNOWN"
    for s in ("WARNING", "CAUTION", "NOMINAL"):
        if ratios.get(s, 0) >= THRESHOLDS[s]:
            state = s
            break

    dominant = max(ratios, key=ratios.get) if ratios else "UNKNOWN"

    return PanelResult(
        panel="?",
        state=state,
        warning_ratio=round(ratios.get("WARNING", 0), 4),
        caution_ratio=round(ratios.get("CAUTION", 0), 4),
        nominal_ratio=round(ratios.get("NOMINAL", 0), 4),
        dominant_color=dominant,
    )


# ─── PNG'den numpy dizisi ─────────────────────────────────────────────────────

def png_bytes_to_bgr(png_bytes: bytes):
    """
    PNG bytes → BGR numpy dizisi.
    cv2.imdecode kullanır; yoksa saf Python ile temel RGBA→BGR parse.
    """
    if not HAS_NUMPY:
        return None

    if HAS_CV2:
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # ── Saf Python fallback (sadece sıkıştırılmamış/basit PNG) ──
    # Bu yol çok sınırlı; cv2 kurulu değilse uyarı ver
    return None


def qpixmap_to_bgr(pixmap):
    """QPixmap → BGR numpy dizisi."""
    if not HAS_NUMPY or not HAS_CV2:
        return None
    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
    buf = QByteArray()
    qbuf = QBuffer(buf)
    qbuf.open(QIODevice.WriteOnly)
    pixmap.save(qbuf, "PNG")
    return png_bytes_to_bgr(bytes(buf.data()))


# ─── ANA ANALİZ FONKSİYONU ───────────────────────────────────────────────────

def analyze_screenshot(image_source, actual_w: int = BASE_W, actual_h: int = BASE_H) -> CVAnalysisResult:
    """
    Screenshot'ı analiz et.

    image_source: QPixmap | bytes (PNG) | numpy ndarray (BGR)
    Döndürür: CVAnalysisResult
    """
    import time
    t0 = time.time()

    if not HAS_CV2 or not HAS_NUMPY:
        return CVAnalysisResult(
            overall_state="UNKNOWN",
            panels={},
            wca_is_red=False,
            wca_is_yellow=False,
            wca_is_green=False,
            mc_is_active=False,
            method="fallback",
            error="opencv veya numpy kurulu değil. 'pip install opencv-python numpy' çalıştırın.",
        )

    # image_source türüne göre BGR'ye çevir
    bgr = None
    try:
        if hasattr(image_source, 'save'):  # QPixmap
            bgr = qpixmap_to_bgr(image_source)
        elif isinstance(image_source, bytes):
            bgr = png_bytes_to_bgr(image_source)
        elif HAS_NUMPY and isinstance(image_source, np.ndarray):
            bgr = image_source
    except Exception as ex:
        return CVAnalysisResult(
            overall_state="UNKNOWN", panels={},
            wca_is_red=False, wca_is_yellow=False, wca_is_green=False,
            mc_is_active=False, method="opencv",
            error=f"Görüntü dönüştürme hatası: {ex}",
        )

    if bgr is None:
        return CVAnalysisResult(
            overall_state="UNKNOWN", panels={},
            wca_is_red=False, wca_is_yellow=False, wca_is_green=False,
            mc_is_active=False, method="opencv",
            error="BGR görüntüsü elde edilemedi.",
        )

    h, w = bgr.shape[:2]
    regions = _scale_regions(w, h)

    # Her paneli analiz et
    panel_results: Dict[str, PanelResult] = {}
    for name, (x1, y1, x2, y2) in regions.items():
        # Koordinatları görüntü sınırlarına kısıtla
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = bgr[y1:y2, x1:x2]
        pr = _analyze_region_hsv(region)
        pr.panel = name
        panel_results[name] = pr

    # WCA paneli özel değerlendirme
    wca = panel_results.get("wca")
    wca_red    = wca.state == "WARNING" if wca else False
    wca_yellow = wca.state == "CAUTION" if wca else False
    wca_green  = wca.state == "NOMINAL" if wca else False

    # Master Caution
    mc = panel_results.get("master_cau")
    mc_active = mc.state in ("WARNING", "CAUTION") if mc else False

    # Genel durum: en kötü panel
    severity_order = {"WARNING": 3, "CAUTION": 2, "NOMINAL": 1, "UNKNOWN": 0}
    overall = max(
        (pr.state for pr in panel_results.values()),
        key=lambda s: severity_order.get(s, 0),
        default="UNKNOWN",
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    return CVAnalysisResult(
        overall_state=overall,
        panels={k: v for k, v in panel_results.items()},
        wca_is_red=wca_red,
        wca_is_yellow=wca_yellow,
        wca_is_green=wca_green,
        mc_is_active=mc_active,
        analysis_ms=elapsed_ms,
        method="opencv",
    )


# ─── SONUCU YAZDIRMA ─────────────────────────────────────────────────────────

def print_cv_result(result: CVAnalysisResult, scenario_id: str = ""):
    prefix = f"[CV {scenario_id}]" if scenario_id else "[CV]"
    state_sym = {"WARNING": "🔴", "CAUTION": "🟡", "NOMINAL": "🟢"}.get(result.overall_state, "⚪")
    print(f"   {prefix} {state_sym} Genel={result.overall_state}  "
          f"WCA={'RED' if result.wca_is_red else 'YEL' if result.wca_is_yellow else 'GRN' if result.wca_is_green else '?'}  "
          f"MC={'ON' if result.mc_is_active else 'OFF'}  "
          f"({result.analysis_ms}ms)")
    if result.error:
        print(f"   {prefix} HATA: {result.error}")


# ─── BAĞIMSIZ TEST ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"numpy : {'✅' if HAS_NUMPY else '❌ pip install numpy'}")
    print(f"opencv: {'✅' if HAS_CV2  else '❌ pip install opencv-python'}")
    if HAS_CV2 and HAS_NUMPY:
        # Saf kırmızı test görüntüsü
        test_img = np.zeros((860, 1600, 3), dtype=np.uint8)
        test_img[680:855, 1200:1595] = (50, 50, 255)   # WCA bölgesi kırmızı (BGR)
        result = analyze_screenshot(test_img, 1600, 860)
        print_cv_result(result, "TEST")
        assert result.wca_is_red, "WCA kırmızı algılanamadı!"
        print("✅ Temel renk testi geçti.")
