from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
from PyQt5.QtGui import QImage


def qimage_to_bgr(qimg: QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w = qimg.width()
    h = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    # RGBA -> BGR
    return arr[:, :, :3][:, :, ::-1].copy()


def grab_widget_bgr(widget) -> np.ndarray:
    pix = widget.grab()
    return qimage_to_bgr(pix.toImage())


def crop_inner(img: np.ndarray, pad: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    pad = max(0, min(pad, h // 4, w // 4))
    return img[pad:h - pad, pad:w - pad].copy()


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def color_masks(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = to_hsv(img_bgr)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    orange = cv2.inRange(hsv, (8, 80, 80), (18, 255, 255))
    yellow = cv2.inRange(hsv, (18, 80, 80), (40, 255, 255))
    green = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))
    blue = cv2.inRange(hsv, (95, 80, 80), (130, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 180), (180, 70, 255))

    return {
        "red": red,
        "orange": orange,
        "yellow": yellow,
        "green": green,
        "blue": blue,
        "white": white,
    }


def dominant_named_color(img_bgr: np.ndarray) -> Tuple[str, Dict[str, int]]:
    masks = color_masks(img_bgr)
    scores = {name: int(mask.sum() // 255) for name, mask in masks.items()}
    best = max(scores, key=scores.get)
    return best, scores


def expected_gauge_angle_deg(value: float, vmin: float = 0.0, vmax: float = 220.0) -> float:
    value = max(vmin, min(vmax, value))
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    # start_deg = 225, span_deg = 270, clockwise drawing
    return (225.0 - 270.0 * t) % 360.0


def detect_speed_needle_angle_deg(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    hsv = to_hsv(img_bgr)

    mask_green = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))
    mask_yellow = cv2.inRange(hsv, (18, 80, 80), (40, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    mask = mask_green | mask_yellow | mask_red1 | mask_red2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cx = w // 2
    cy = int(h * 0.78)  # widget çizim merkezine yakın

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise AssertionError("SpeedGauge ibresi için renkli piksel bulunamadı.")

    # Merkezden en uzak renkli pikseli ibre ucu kabul ediyoruz
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    i = int(np.argmax(d2))
    tip_x, tip_y = int(xs[i]), int(ys[i])

    dx = tip_x - cx
    dy = cy - tip_y
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return angle


def angular_error_deg(a: float, b: float) -> float:
    return min((a - b) % 360.0, (b - a) % 360.0)


def assert_speed_needle_angle(img_bgr: np.ndarray, expected_value: float, tol_deg: float = 8.0):
    got = detect_speed_needle_angle_deg(img_bgr)
    exp = expected_gauge_angle_deg(expected_value)
    err = angular_error_deg(got, exp)
    assert err <= tol_deg, (
        f"IAS ibre açısı beklenenden sapmış. "
        f"beklenen={exp:.2f}°, bulunan={got:.2f}°, hata={err:.2f}°"
    )


def detect_bar_fill_ratio(img_bgr: np.ndarray) -> float:
    inner = crop_inner(img_bgr, pad=2)
    hsv = to_hsv(inner)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Beyaz border ve siyah zemin dışarıda kalır, renkli dolgu kalır
    colored = ((sat > 60) & (val > 60)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(colored > 0)
    if len(xs) == 0:
        return 0.0

    y_top = int(ys.min())
    y_bottom = int(ys.max())
    fill_h = y_bottom - y_top + 1
    h = inner.shape[0]
    return fill_h / max(1, h)


def assert_bar_fill_close(
    img_bgr: np.ndarray,
    expected_ratio: float,
    tol: float = 0.12,
):
    got = detect_bar_fill_ratio(img_bgr)
    expected_ratio = max(0.0, min(1.0, expected_ratio))
    err = abs(got - expected_ratio)
    assert err <= tol, (
        f"Bar fill ratio beklenenden sapmış. "
        f"beklenen={expected_ratio:.3f}, bulunan={got:.3f}, hata={err:.3f}"
    )


def classify_bar_color(img_bgr: np.ndarray) -> str:
    inner = crop_inner(img_bgr, pad=2)
    name, _ = dominant_named_color(inner)
    return name


def assert_bar_color_in(img_bgr: np.ndarray, expected_names):
    if isinstance(expected_names, str):
        expected_names = {expected_names}
    else:
        expected_names = set(expected_names)

    got = classify_bar_color(img_bgr)
    assert got in expected_names, f"Bar rengi beklenen sınıfta değil. beklenen={expected_names}, bulunan={got}"


def detect_invalid_cross_strength(img_bgr: np.ndarray) -> Tuple[int, int]:
    inner = crop_inner(img_bgr, pad=2)
    masks = color_masks(inner)
    red = masks["red"]

    h, w = red.shape
    diag1 = 0
    diag2 = 0

    for x in range(0, w):
        y1 = int((h - 1) * x / max(1, w - 1))
        y2 = int((h - 1) * (1.0 - x / max(1, w - 1)))

        p1 = red[max(0, y1 - 2):min(h, y1 + 3), max(0, x - 2):min(w, x + 3)]
        p2 = red[max(0, y2 - 2):min(h, y2 + 3), max(0, x - 2):min(w, x + 3)]

        if np.count_nonzero(p1) > 0:
            diag1 += 1
        if np.count_nonzero(p2) > 0:
            diag2 += 1

    return diag1, diag2


def assert_invalid_cross_present(img_bgr: np.ndarray, min_ratio: float = 0.35):
    diag1, diag2 = detect_invalid_cross_strength(img_bgr)
    w = img_bgr.shape[1]
    assert diag1 >= int(w * min_ratio) and diag2 >= int(w * min_ratio), (
        f"Invalid kırmızı X yeterince güçlü görünmüyor. diag1={diag1}, diag2={diag2}, width={w}"
    )


def classify_wca_panel_color(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    # Başlık ve border etkisini azaltmak için orta bölgeden kes
    roi = img_bgr[int(h * 0.10):int(h * 0.90), int(w * 0.05):int(w * 0.95)]
    name, _ = dominant_named_color(roi)

    # WCA için pratikte red / yellow / green bekliyoruz
    if name == "orange":
        return "yellow"
    return name