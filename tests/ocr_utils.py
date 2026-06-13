"""
TUSAS TestLab — OCR Utilities
=============================
Ekranda RENDER EDİLMİŞ pikselleri okur (Qt veri modelini DEĞİL).

Neden önemli:
  Eski testler `widget.lbl_val.text()` ile veri modelini okuyordu — yani
  "ekranda ne yazıyor" değil "kodda ne var" kontrol ediliyordu. Paint/render
  hatası (yanlış konum, kırpılma, üst üste binme, yanlış font) asla yakalanamazdı.

  Bu modül screenshot'tan ilgili bölgeyi kırpar, koyu kokpit temasına göre
  ön-işler (gri tonlama → upscale → invert → Otsu eşikleme) ve tesseract ile
  OCR yapar. Böylece "gerçekten ekranda görünen" sayı/birim doğrulanır.

Bağımlılık: pytesseract + tesseract-ocr (sistem), opencv, numpy, Pillow.
Hiçbiri yoksa fonksiyonlar (ok=False, reason="ocr_unavailable") döner ve
çağıran taraf OCR kontrolünü atlar (testi düşürmeden).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import pytesseract
    # tesseract binary gerçekten var mı?
    pytesseract.get_tesseract_version()
    HAS_TESS = True
except Exception:
    HAS_TESS = False

OCR_AVAILABLE = HAS_CV2 and HAS_TESS


@dataclass
class OcrResult:
    ok: bool
    raw_text: str = ""
    value: Optional[float] = None      # sayı parse edilebildiyse
    reason: str = ""                   # ok=False ise neden


# ─── ÖN İŞLEME ────────────────────────────────────────────────────────────────

def _preprocess(crop_rgb: np.ndarray, scale: int = 7) -> np.ndarray:
    """
    Koyu zemin + açık renk monospace (Consolas) yazı için OCR ön işleme.
      1. Gri tonlama
      2. Büyütme (küçük etiketler için kritik)
      3. Invert (beyaz yazı/koyu zemin → koyu yazı/beyaz zemin)
      4. Otsu eşikleme (binarize)
      5. Morfolojik kapama — Consolas'ın noktalı sıfırını kapatıp 0↔6 karışmasını azaltır
      6. Beyaz kenarlık (tesseract için quiet-zone)

    Bu kombinasyon bu fontta deneysel olarak en yüksek doğruluğu verdi.
    """
    if crop_rgb.ndim == 3:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_rgb

    mean_v = float(gray.mean())
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if mean_v < 128:           # koyu zemin (kokpit) → yazıyı koyulaştır
        gray = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    binary = cv2.copyMakeBorder(binary, 25, 25, 25, 25,
                                cv2.BORDER_CONSTANT, value=255)
    return binary


def _crop_with_pad(pil_img, bbox: Tuple[int, int, int, int], pad: int = 6):
    """bbox (x, y, w, h) + padding ile PIL crop → numpy RGB."""
    x, y, w, h = bbox
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(pil_img.width, x + w + pad)
    bottom = min(pil_img.height, y + h + pad)
    crop = pil_img.crop((left, top, right, bottom)).convert("RGB")
    return np.array(crop)


# ─── SAYI OKUMA ───────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def ocr_number(pil_img, bbox: Tuple[int, int, int, int]) -> OcrResult:
    """
    Verilen bbox bölgesindeki RENDER EDİLMİŞ sayıyı OCR ile oku.
    Döndürür: OcrResult(value=float | None)
    """
    if not OCR_AVAILABLE:
        return OcrResult(ok=False, reason="ocr_unavailable")
    if pil_img is None or bbox is None:
        return OcrResult(ok=False, reason="no_image_or_bbox")

    try:
        crop = _crop_with_pad(pil_img, bbox, pad=6)
        if crop.size == 0 or min(crop.shape[:2]) < 2:
            return OcrResult(ok=False, reason="empty_crop")

        binary = _preprocess(crop)
        # oem 1 (LSTM), psm 7: tek satır. Rakam + işaret + ondalık + birim harfleri.
        cfg = ("--oem 1 --psm 7 -c "
               "tessedit_char_whitelist=0123456789.,-%CFKLBSPIRANGVZacfompsiz° ")
        raw = pytesseract.image_to_string(binary, config=cfg).strip()
        raw = raw.replace("O", "0").replace("o", "0").replace("l", "1")

        m = _NUM_RE.search(raw.replace(" ", ""))
        if not m:
            return OcrResult(ok=True, raw_text=raw, value=None,
                             reason="no_number_parsed")
        num = float(m.group(0).replace(",", "."))
        return OcrResult(ok=True, raw_text=raw, value=num)
    except Exception as e:
        return OcrResult(ok=False, reason=f"ocr_error: {e}")


def ocr_text(pil_img, bbox: Tuple[int, int, int, int]) -> OcrResult:
    """Verilen bölgedeki serbest metni OCR ile oku (birim/etiket için)."""
    if not OCR_AVAILABLE:
        return OcrResult(ok=False, reason="ocr_unavailable")
    if pil_img is None or bbox is None:
        return OcrResult(ok=False, reason="no_image_or_bbox")
    try:
        crop = _crop_with_pad(pil_img, bbox, pad=6)
        if crop.size == 0 or min(crop.shape[:2]) < 2:
            return OcrResult(ok=False, reason="empty_crop")
        binary = _preprocess(crop)
        raw = pytesseract.image_to_string(binary, config="--psm 7").strip()
        return OcrResult(ok=True, raw_text=raw)
    except Exception as e:
        return OcrResult(ok=False, reason=f"ocr_error: {e}")
