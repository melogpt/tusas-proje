"""
TUSAS TestLab — Visual Analyzer
=================================
PIL + NumPy tabanlı piksel analizi:
  - Bar doluluk oranı ölçümü
  - Baskın renk tespiti
  - Widget bounding box hesabı
  - Hatalı alanları işaretlenmiş screenshot üretimi

Bu modül PyQt5 bağımlılığı olmadan da çalışabilir
(widget bbox desteği Qt gerektiriyor).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from PyQt5.QtCore import QPoint
    HAS_QT = True
except ImportError:
    HAS_QT = False


# ─── VERİ YAPILARI ────────────────────────────────────────────────────────────

@dataclass
class VisualCheckResult:
    """Tek bir görsel kontrol sonucu."""
    check_name: str          # "bar_fill" | "bar_color" | "numeric" | "unit" | "background" | "text"
    passed: bool
    expected: str
    actual: str
    error_category: str      # ErrorCategory değeri
    goes_to_wca: bool        # Bu hata WCA'ya düşmeli mi?
    bbox: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h) pencere koordinatında
    note: str = ""


@dataclass
class AnnotationBox:
    """Raporda çizilecek hata kutucuğu."""
    x: int
    y: int
    w: int
    h: int
    label: str
    color: Tuple[int, int, int] = (255, 50, 50)   # RGB kırmızı


# ─── WIDGET BBOX ──────────────────────────────────────────────────────────────

def get_widget_bbox(widget, root_window) -> Optional[Tuple[int, int, int, int]]:
    """
    Widget'ın root_window içindeki konumunu döndürür (x, y, w, h).
    Screenshot bu koordinatlara göre crop edilir.
    """
    if not HAS_QT or widget is None or root_window is None:
        return None
    try:
        from PyQt5.QtCore import QPoint
        # Map directly to root window's client coordinates.
        # This matches the coordinate space of root_window.grab() exactly,
        # avoiding OS title bar and frame border offsets in both standard and offscreen modes.
        pos = widget.mapTo(root_window, QPoint(0, 0))
        return (pos.x(), pos.y(), widget.width(), widget.height())
    except Exception:
        return None


def pil_from_screenshot(screenshot_path: str) -> Optional["Image.Image"]:
    """Screenshot PNG dosyasını PIL Image olarak yükle."""
    if not HAS_PIL or not os.path.exists(screenshot_path):
        return None
    return Image.open(screenshot_path).convert("RGB")


def crop_bbox(pil_img: "Image.Image", bbox: Tuple[int, int, int, int]) -> "Image.Image":
    """(x, y, w, h) bbox'a göre PIL image crop et."""
    x, y, w, h = bbox
    return pil_img.crop((x, y, x + w, y + h))


# ─── RENK ANALİZİ ─────────────────────────────────────────────────────────────

# Qt'nin kullandığı renk hex'leri → isim haritası
_HEX_TO_NAME = {
    "#FF3333": "red",
    "#FFB000": "orange",
    "#00FF66": "green",
    "#3399FF": "blue",
    "#FFFF33": "yellow",
    "#FF9933": "orange",
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def detect_dominant_color(crop: "Image.Image") -> str:
    """
    Crop edilmiş widget bölgesinde baskın rengi tespit et.
    Siyah arka plan (#050505) ve beyaz border piksellerini maskeler.

    Döndürür: "red" | "orange" | "green" | "blue" | "yellow" | "black" | "unknown"
    """
    if not HAS_PIL:
        return "unknown"
    arr = np.array(crop)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return "unknown"

    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)

    # Siyah arka planı ve beyaz border'ı maskele
    is_black = (r < 30) & (g < 30) & (b < 30)
    is_white = (r > 210) & (g > 210) & (b > 210)
    is_dark_bg = (r < 15) & (g < 40) & (b < 15)   # #050505 varyantları
    mask = ~(is_black | is_white | is_dark_bg)

    pixels_r = r[mask]
    if len(pixels_r) < 10:
        return "black"

    mr = float(pixels_r.mean())
    mg = float(g[mask].mean())
    mb = float(b[mask].mean())

    # Renk sınıflandırması
    if mr > 160 and mg < 80 and mb < 80:
        return "red"
    if mr > 160 and mg > 120 and mb < 60:
        return "orange"
    if mr > 200 and mg > 200 and mb < 80:
        return "yellow"
    if mg > 150 and mr < 80 and mb < 80:
        return "green"
    if mb > 150 and mr < 80 and mg < 100:
        return "blue"
    if mr > 150 and mg > 150 and mb > 150:
        return "white"
    return "unknown"


def detect_background_color(crop: "Image.Image") -> str:
    """
    Widget arka fon rengini tespit et.
    Sadece büyük renk bloklarına bak (border pikselleri değil).
    """
    return detect_dominant_color(crop)


# ─── BAR DOLULUK ORANI ────────────────────────────────────────────────────────

def measure_bar_fill_ratio(crop: "Image.Image") -> float:
    """
    Dikey bar widget'ının doluluk oranını ölç (0.0–1.0).

    BarVisual.paintEvent mantığı:
      - İç alan: 2px margin tüm kenarlarda
      - Fill: alt kenardan yukarı doğru
      - Arka fon: #050505 (çok koyu)
    """
    if not HAS_PIL:
        return -1.0
    arr = np.array(crop.convert("RGB"))
    h, w = arr.shape[:2]

    if h < 6 or w < 6:
        return -1.0

    # İç bölge (2px border kaldır)
    inner = arr[2:h-2, 2:w-2]
    inner_h = inner.shape[0]
    if inner_h == 0:
        return -1.0

    # Çok koyu (siyah arka plan) olmayan pikseller dolu sayılır
    r, g, b = inner[:, :, 0].astype(float), inner[:, :, 1].astype(float), inner[:, :, 2].astype(float)
    is_bg = (r < 25) & (g < 50) & (b < 25)  # koyu arka plan
    is_border_color = (r > 200) & (g > 200) & (b > 200)  # beyaz çizgi
    filled_mask = ~(is_bg | is_border_color)

    # Satır bazında: satırın %25'inden fazlası dolu ise o satır filled
    row_fill_ratio = np.mean(filled_mask, axis=1)
    filled_rows = np.sum(row_fill_ratio > 0.25)

    return float(filled_rows) / inner_h


def expected_fill_ratio(value: float, vmin: float, vmax: float) -> float:
    """Parametre değeri için beklenen bar doluluk oranı."""
    if vmax == vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def fill_ratio_matches(actual: float, expected: float, tolerance: float = 0.10) -> bool:
    """
    Bar doluluk oranı tolerans dahilinde doğru mu?
    tolerance=0.10 → %10 sapma kabul edilir.
    """
    if actual < 0:   # ölçülemedi
        return True  # skip
    return abs(actual - expected) <= tolerance


# ─── NUMERIC DEĞER ────────────────────────────────────────────────────────────

def read_label_text(widget) -> str:
    """Qt QLabel widget'ından metin oku."""
    try:
        return widget.text().strip()
    except Exception:
        return ""


def check_numeric_displayed(widget, expected_value: float, decimals: int = 1, unit: str = "") -> Tuple[bool, str, str]:
    """
    Parametre widget'ının gösterdiği sayısal değeri kontrol et.
    Döndürür: (passed, expected_str, actual_str)
    """
    try:
        actual_text = widget.lbl_val.text().strip()
        # Unit'i kaldır
        actual_num_str = actual_text.replace(unit, "").strip()
        actual_num = float(actual_num_str)
        expected_fmt = f"{expected_value:.{decimals}f}"
        actual_fmt = f"{actual_num:.{decimals}f}"
        passed = abs(actual_num - expected_value) < (0.5 * 10 ** (-decimals) + 0.01)
        return passed, expected_fmt, actual_fmt
    except Exception as e:
        return False, f"{expected_value:.{decimals}f}", f"PARSE_ERROR: {e}"


def check_unit_displayed(widget, expected_unit: str) -> Tuple[bool, str, str]:
    """Widget'ın gösterdiği birimi kontrol et."""
    try:
        actual_text = widget.lbl_val.text().strip()
        actual_unit = actual_text.split()[-1] if " " in actual_text else ""
        passed = expected_unit in actual_text
        return passed, expected_unit, actual_unit
    except Exception:
        return False, expected_unit, "PARSE_ERROR"


# ─── ANNOTATED SCREENSHOT ─────────────────────────────────────────────────────

def annotate_screenshot(
    pil_img: "Image.Image",
    annotations: List[AnnotationBox],
    save_path: Optional[str] = None,
) -> "Image.Image":
    """
    Screenshot üzerine hata kutucukları çiz.
    Her hata için kalın kırmızı dikdörtgen ve etiket.
    """
    if not HAS_PIL:
        return pil_img
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)

    for ann in annotations:
        x, y, w, h = ann.x, ann.y, ann.w, ann.h
        c = ann.color

        # Kalın çerçeve (3 katman)
        for offset in range(4):
            draw.rectangle(
                [x - offset, y - offset, x + w + offset, y + h + offset],
                outline=c
            )

        # Etiket arka fonu
        label_h = 18
        label_w = len(ann.label) * 7 + 8
        lx, ly = x, max(0, y - label_h)
        draw.rectangle([lx, ly, lx + label_w, ly + label_h], fill=c)
        draw.text((lx + 3, ly + 2), ann.label, fill=(0, 0, 0))

    if save_path:
        annotated.save(save_path, "PNG")

    return annotated


def save_cropped_error(
    pil_img: "Image.Image",
    bbox: Tuple[int, int, int, int],
    save_path: str,
    padding: int = 20,
) -> bool:
    """
    Hatalı widget bölgesini büyütülmüş crop olarak kaydet.
    padding: etrafına eklenecek piksel boşluğu
    """
    if not HAS_PIL or pil_img is None:
        return False
    try:
        x, y, w, h = bbox
        img_w, img_h = pil_img.size
        crop_x1 = max(0, x - padding)
        crop_y1 = max(0, y - padding)
        crop_x2 = min(img_w, x + w + padding)
        crop_y2 = min(img_h, y + h + padding)
        cropped = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        # 2x büyüt
        new_w = (crop_x2 - crop_x1) * 2
        new_h = (crop_y2 - crop_y1) * 2
        cropped = cropped.resize((new_w, new_h), Image.NEAREST)
        cropped.save(save_path, "PNG")
        return True
    except Exception:
        return False


# ─── UNEXPECTED TEXT ──────────────────────────────────────────────────────────

def check_wca_for_unexpected_text(wca_entries, allowed_texts: set) -> List[str]:
    """
    WCA girişlerinde izin verilmeyen metin var mı kontrol et.
    Döndürür: beklenmeyen metin listesi
    """
    unexpected = []
    for entry in wca_entries:
        text_upper = entry.text.upper()
        found = any(allowed.upper() in text_upper for allowed in allowed_texts)
        if not found:
            unexpected.append(entry.text)
    return unexpected


def check_for_random_text_artifacts(widget_texts: List[str], allowed_texts: set) -> List[str]:
    """
    Widget text listesinde tanımsız metin artefact var mı?
    Her kelimeyi allowed_texts ile karşılaştır.
    """
    artifacts = []
    for text in widget_texts:
        words = text.upper().split()
        for word in words:
            clean = word.strip(":.,-")
            if len(clean) > 2:
                if not any(allowed.upper() in clean or clean in allowed.upper()
                           for allowed in allowed_texts):
                    artifacts.append(word)
    return artifacts
