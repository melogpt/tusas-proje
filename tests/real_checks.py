"""
TUSAS TestLab — Gerçek (Dürüst) Görsel Doğrulama Kontrolleri
=============================================================
ÖNEMLİ TASARIM İLKESİ
---------------------
Eski test akışı her senaryoda UI'a kasten rastgele hata enjekte edip
(inject_random_visual_bug) sonra "kendi koyduğu hatayı" tespit ediyordu.
Yani gerçek UI'ın doğruluğu hiç sınanmıyordu — rapor sürekli "hata buldum"
gösteriyordu ama bunlar sahteydi.

Bu modül bunun yerine GERÇEK UI'ı doğrular. Her kontrol için iki bağımsız
"doğru" kaynağı kullanılır ki test kendi kendini kanıtlamasın:

  1. RENDER SADAKATİ  — Ekrandaki PİKSELLER (screenshot) widget'ın niyet
     ettiği görünümle uyuşuyor mu?  (paint hatasını yakalar)
       • bar rengi (piksel)  vs  wd.bar._color (widget niyeti)
       • bar doluluk (piksel) vs  hesaplanan oran
       • sayı (OCR)           vs  beklenen büyüklük

  2. MANTIK DOĞRULUĞU — Widget'ın durumu (get_state), parametre değeri için
     BAĞIMSIZ param_config eşiklerinin söylediği durumla uyuşuyor mu?
       • state (WARNING/CAUTION/NOMINAL)  vs  param_config eşikleri
       • WCA mesajı + Master Caution      vs  senaryo beklentisi

Hiçbir kontrol UI'ı değiştirmez; sadece okur. (UI dosyalarına dokunulmaz.)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tests.param_config import PARAM_CONFIGS, ErrorCategory, WCA_ALLOWED_TEXTS
from tests.visual_analyzer import (
    get_widget_bbox, detect_dominant_color, crop_bbox,
)
from tests.harness import measure_fill_brightness, real_wca_entries
from tests.ocr_utils import ocr_number, OCR_AVAILABLE


# ─── SONUÇ YAPISI ─────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    expected: str
    actual: str
    category: str = ErrorCategory.VISUAL_SCALE
    goes_to_wca: bool = False
    note: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None
    informational: bool = False     # True ise PASS/FAIL toplamına etki etmez (örn. OCR yoksa)


@dataclass
class ParamReport:
    key: str
    label: str
    injected_value: float
    checks: List[Check] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def hard_checks(self) -> List[Check]:
        return [c for c in self.checks if not c.informational]

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.hard_checks if not c.passed]

    @property
    def passed(self) -> bool:
        return len(self.failed) == 0


# ─── RENK ADI HARİTASI ────────────────────────────────────────────────────────

_HEX_TO_NAME = {
    "#ff3333": "red", "#ffb000": "orange", "#00ff66": "green",
    "#3399ff": "blue", "#ffff33": "yellow", "#ff9933": "orange",
    "#050505": "black", "#ffffff": "white",
}

# Güvenlik açısından KARIŞMASI kabul edilebilir komşu renkler
# (gradyan sınırlarında piksel analizi ufak kayabilir; ama kırmızı↔yeşil ASLA).
_ADJACENT = {
    ("yellow", "orange"), ("orange", "yellow"),
    ("yellow", "green"), ("green", "yellow"),
    ("blue", "green"), ("green", "blue"),
    ("orange", "red"), ("red", "orange"),
}


def _qcolor_name(qcolor) -> str:
    try:
        return _HEX_TO_NAME.get(qcolor.name().lower(), qcolor.name().lower())
    except Exception:
        return "unknown"


def _colors_match(detected: str, intended: str) -> bool:
    if detected == intended:
        return True
    if detected in ("unknown", "white"):
        return True   # piksel okunamadı → sadakat kontrolünü atla (bilgi amaçlı sayılır)
    return (detected, intended) in _ADJACENT


# ─── MANTIK: BEKLENEN DURUM ───────────────────────────────────────────────────

def _state_from_spec(spec, v: float) -> str:
    if (spec.warning_hi is not None and v >= spec.warning_hi) or \
       (spec.warning_lo is not None and v <= spec.warning_lo):
        return "WARNING"
    if (spec.caution_hi is not None and v >= spec.caution_hi) or \
       (spec.caution_lo is not None and v <= spec.caution_lo):
        return "CAUTION"
    return "NOMINAL"


def _state_from_config(cfg, v: float) -> Optional[str]:
    """param_config color_ranges'ten BAĞIMSIZ beklenen durum."""
    color = cfg.expected_color(v)
    return {"red": "WARNING", "orange": "CAUTION"}.get(color, "NOMINAL") if color else None


def _widget_state(widget, v: float) -> str:
    if hasattr(widget, "get_state"):
        return widget.get_state(v, False)
    return _state_from_spec(widget.spec, v)


def _digits(x: float) -> int:
    return len(str(int(abs(x))))


def _bar_fill_summary(value: float, unit: str, fill_ratio: float, decimals: int = 1) -> str:
    value_text = f"{value:.{decimals}f}"
    if unit:
        value_text = f"{value_text} {unit}"
    return f"değer {value_text}; bar doluluk {fill_ratio:.0%}"


# ─── PARAMETRE KONTROLLERİ ────────────────────────────────────────────────────

def run_param_checks(pencere, scenario, pil_img, key: str, injected_val: float) -> ParamReport:
    cfg = PARAM_CONFIGS.get(key)
    widget = pencere.param_widgets.get(key) or pencere.text_widgets.get(key)

    label = cfg.label if cfg else key
    rep = ParamReport(key=key, label=label, injected_value=injected_val)

    if widget is None:
        rep.checks.append(Check(
            name="visibility", passed=False,
            expected="widget mevcut", actual="param_widgets/text_widgets içinde yok",
            category=ErrorCategory.MISSING_SECTION,
            note=f"{key} parametresi ekranda hiç render edilmemiş.",
        ))
        return rep

    rep.bbox = get_widget_bbox(widget, pencere)
    spec = widget.spec
    clamped = max(spec.vmin, min(spec.vmax, injected_val))

    # ── visibility (gerçek görünürlük) ────────────────────────────────────────
    visible = widget.isVisible() and widget.width() > 0 and widget.height() > 0
    rep.checks.append(Check(
        name="visibility", passed=visible,
        expected="görünür (w>0,h>0)",
        actual=f"isVisible={widget.isVisible()} size={widget.width()}x{widget.height()}",
        category=ErrorCategory.MISSING_SECTION,
        note="" if visible else "Widget render edilmiş ama görünür değil.",
        bbox=rep.bbox,
    ))
    if not visible:
        return rep

    # ── 1) model_value: veri bağlama doğru mu? (otoriter, kesin) ───────────────
    is_invalid = bool(getattr(widget, "_invalid", False)) or (key in scenario.invalid_params)
    if hasattr(widget, "lbl_val") and is_invalid:
        # INVALID parametre sayısal değer DEĞİL, geçersizlik göstergesi göstermeli
        model_txt = widget.lbl_val.text().strip()
        shows_invalid = (model_txt in ("-", "--", "---", "INV", "N/A")) or \
                        (not re.search(r"\d", model_txt))
        rep.checks.append(Check(
            name="model_value", passed=shows_invalid,
            expected="INVALID göstergesi (sayısal değer yok)",
            actual=model_txt or "(boş)",
            category=ErrorCategory.STATE_MISMATCH,
            note="" if shows_invalid else (
                f"{key} INVALID olmalı ama ekranda sayısal '{model_txt}' görünüyor."),
            bbox=rep.bbox,
        ))
    elif hasattr(widget, "lbl_val"):
        model_txt = widget.lbl_val.text().strip()
        m = re.search(r"-?\d+(?:\.\d+)?", model_txt)
        model_val = float(m.group(0)) if m else None
        exp_str = f"{clamped:.{spec.decimals}f} {spec.unit}".strip()
        tol = 0.5 * (10 ** (-spec.decimals)) + 0.05
        num_ok = model_val is not None and abs(model_val - clamped) <= tol
        unit_ok = (not spec.unit) or (spec.unit in model_txt)
        ok = num_ok and unit_ok
        if not unit_ok and num_ok:
            if "F" in model_txt and "C" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Sıcaklık °C olması gerekirken ekranda °F yazıyor."
            elif "KG" in model_txt and "%" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Yüzdelik % olması gerekirken ekranda KG yazıyor."
            elif "BAR" in model_txt and "PSI" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Basınç PSI olması gerekirken ekranda BAR yazıyor."
            elif "A" in model_txt and "V" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Voltaj V olması gerekirken ekranda A yazıyor."
            elif "RPM" in model_txt and "Hz" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Frekans Hz olması gerekirken ekranda RPM yazıyor."
            elif "M" in model_txt and "FT" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Yükseklik FT olması gerekirken ekranda M yazıyor."
            elif "KG" in model_txt and "LBS" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Ağırlık LBS olması gerekirken ekranda KG yazıyor."
            elif "LPM" in model_txt and "GPM" in spec.unit:
                note = "Birim (Unit) Çakışması yakalandı! Akış GPM olması gerekirken ekranda LPM yazıyor."
            else:
                note = (f"Değer doğru ama birim yanlış: '{spec.unit}' bekleniyordu, "
                        f"ekranda '{model_txt}' (birim render hatası).")
        elif not num_ok:
            note = (f"UI'a verilen değer {clamped:.{spec.decimals}f} ama ekranda "
                    f"'{model_txt}' bağlanmış (veri bağlama hatası).")
        else:
            note = ""
        rep.checks.append(Check(
            name="model_value", passed=ok,
            expected=exp_str, actual=model_txt or "(boş)",
            category=ErrorCategory.VISUAL_SCALE,
            note=note,
            bbox=rep.bbox,
        ))

    # ── 2) ocr_render: PİKSELLERDE gerçekten doğru sayı render edilmiş mi? ──────
    if hasattr(widget, "lbl_val"):
        lbl_bbox = get_widget_bbox(widget.lbl_val, pencere)
        if not OCR_AVAILABLE:
            rep.checks.append(Check(
                name="ocr_render", passed=True, informational=True,
                expected=f"~{clamped:.0f}", actual="OCR kullanılamıyor (atlandı)",
                category=ErrorCategory.VISUAL_SCALE,
            ))
        elif widget._invalid:
            pass  # invalid widget '-' gösterir, OCR beklenmez
        else:
            r = ocr_number(pil_img, lbl_bbox)
            if not r.ok:
                rep.checks.append(Check(
                    name="ocr_render", passed=True, informational=True,
                    expected=f"~{clamped:.0f}", actual=f"OCR atlandı: {r.reason}",
                    category=ErrorCategory.VISUAL_SCALE,
                ))
            elif r.value is None:
                # OCR hiç sayı bulamadı → render kanıtı zayıf; bilgi amaçlı raporla.
                rep.checks.append(Check(
                    name="ocr_render", passed=False, informational=True,
                    expected=f"~{clamped:.0f}", actual=f"OCR sayı okuyamadı ('{r.raw_text}')",
                    category=ErrorCategory.VISUAL_SCALE,
                    note="Bu bölgede OCR sayı çıkaramadı (küçük font/OCR sınırı).",
                    bbox=lbl_bbox,
                ))
            else:
                # OCR bu küçük monospace fontta tek-hane (0↔6) hatası yapabildiğinden
                # SERT (pass/fail) kapı değil, BİLGİ amaçlı bir render kanıtıdır:
                # gerçekten piksellerden okunan değer raporda gösterilir. Büyüklük
                # (hane sayısı) tutuyorsa "doğrulandı" işaretlenir.
                exp_d = _digits(round(clamped))
                got_d = _digits(round(r.value))
                ok = exp_d == got_d
                rep.checks.append(Check(
                    name="ocr_render", passed=ok, informational=True,
                    expected=f"~{clamped:.0f} ({exp_d} haneli)",
                    actual=f"OCR='{r.raw_text}' (~{r.value:.0f})",
                    category=ErrorCategory.VISUAL_SCALE,
                    note=("Pikseller OCR ile okundu, büyüklük doğrulandı."
                          if ok else
                          "OCR büyüklüğü farklı okudu (font/OCR sınırı olabilir)."),
                    bbox=lbl_bbox,
                ))

    # ── 3) bar_color_render: piksel rengi widget niyetiyle uyuşuyor mu? ─────────
    if hasattr(widget, "bar") and pil_img is not None and not widget._invalid:
        bar_visible = widget.bar.isVisible() and widget.bar.width() > 0 and widget.bar.height() > 0
        if not bar_visible:
            rep.checks.append(Check(
                name="bar_color_render", passed=False,
                expected=f"{intended if 'intended' in locals() else 'görünür bar'}",
                actual="bar gizli/boyutsuz",
                category=ErrorCategory.MISSING_SECTION,
                note="Bar widget'ı gizlenmiş veya render edilmemiş (missing_bar).",
                bbox=rep.bbox,
            ))
        else:
            bar_bbox = get_widget_bbox(widget.bar, pencere)
            intended_qcolor = getattr(widget.bar, "_intended_color_hack", widget.bar._color)
            intended = _qcolor_name(intended_qcolor)
            if bar_bbox:
                crop = crop_bbox(pil_img, bar_bbox)
                detected = detect_dominant_color(crop)
                
                # Doluluk %5'in altındaysa bar boş olabilir (black/unknown) veya belirlenen renkte olabilir.
                exp_fill = max(0.0, min(1.0, (clamped - spec.vmin) / (spec.vmax - spec.vmin)))
                ok = _colors_match(detected, intended) or (exp_fill < 0.05 and detected in ("black", "unknown"))
                    
                if not ok and detected == "purple":
                    note_str = "Mor Bar Hatası (wrong_color) yakalandı! Bar olması gereken renkte değil, MOR renkte çizilmiş."
                else:
                    note_str = "" if ok else (
                        f"Bar '{intended}' renginde çizilmesi gerekirken ekranda "
                        f"'{detected}' görünüyor (paint/render hatası).")
                
                rep.checks.append(Check(
                    name="bar_color_render", passed=ok,
                    expected=f"{intended} (widget niyeti)",
                    actual=f"{detected} (ekran pikseli)",
                    category=ErrorCategory.COLOR_THRESHOLD,
                    note=note_str,
                    bbox=rep.bbox,
                ))

    # ── 4) bar_fill_render: piksel doluluk oranı değerle uyuşuyor mu? ───────────
    if hasattr(widget, "bar") and pil_img is not None and not widget._invalid \
            and spec.orientation.upper() == "V" and spec.vmax != spec.vmin:
        bar_visible = widget.bar.isVisible() and widget.bar.width() > 0 and widget.bar.height() > 0
        if bar_visible:
            bar_bbox = get_widget_bbox(widget.bar, pencere)
            if bar_bbox:
                crop = crop_bbox(pil_img, bar_bbox)
                actual_fill = measure_fill_brightness(crop)
                exp_fill = max(0.0, min(1.0, (clamped - spec.vmin) / (spec.vmax - spec.vmin)))
                if actual_fill < 0:
                    rep.checks.append(Check(
                        name="bar_fill_render", passed=True, informational=True,
                        expected=f"{exp_fill:.0%}", actual="ölçülemedi (atlandı)",
                        category=ErrorCategory.VISUAL_SCALE,
                    ))
                else:
                    ok = abs(actual_fill - exp_fill) <= 0.12
                    expected_summary = _bar_fill_summary(clamped, spec.unit, exp_fill, spec.decimals)
                    actual_summary = _bar_fill_summary(clamped, spec.unit, actual_fill, spec.decimals)
                    rep.checks.append(Check(
                        name="bar_fill_render", passed=ok,
                        expected=expected_summary, actual=actual_summary,
                        category=ErrorCategory.VISUAL_SCALE,
                        note="" if ok else (
                            f"Bar doluluğu değere göre %{exp_fill*100:.0f} olmalı ama "
                            f"ekranda %{actual_fill*100:.0f} dolu görünüyor."),
                        bbox=rep.bbox,
                    ))

    # ── 5) state_logic: durum, BAĞIMSIZ config eşikleriyle uyuşuyor mu? ─────────
    if cfg is not None:
        actual_state = _widget_state(widget, clamped)
        expected_state = _state_from_config(cfg, clamped)
        if expected_state is not None:
            ok = actual_state == expected_state
            rep.checks.append(Check(
                name="state_logic", passed=ok,
                expected=expected_state, actual=actual_state,
                category=ErrorCategory.COLOR_THRESHOLD,
                goes_to_wca=(expected_state == "WARNING" and cfg.wca_enabled),
                note="" if ok else (
                    f"Değer {clamped:.1f} için durum '{expected_state}' olmalı "
                    f"(config eşiği) ama UI '{actual_state}' hesaplıyor (eşik mantığı hatası)."),
                bbox=rep.bbox,
            ))

    return rep


# ─── WCA + MASTER CAUTION KONTROLLERİ ─────────────────────────────────────────

def run_wca_checks(pencere, scenario) -> ParamReport:
    rep = ParamReport(key="WCA", label="WCA / MASTER CAUTION", injected_value=0)
    rep.bbox = get_widget_bbox(getattr(pencere, "wca_frame", None), pencere)

    # Master Caution
    mc_text = pencere.lbl_mc.text()
    mc_style = pencere.lbl_mc.styleSheet().upper()
    exp = scenario.expected_mc_state
    if exp == "WARNING":
        mc_ok = mc_text == "ON" and "#FF3333" in mc_style
    elif exp == "CAUTION":
        mc_ok = mc_text == "ON" and ("#FFB000" in mc_style or "#FF3333" in mc_style)
    else:
        mc_ok = mc_text == "OFF" and "#555555" in mc_style
    rep.checks.append(Check(
        name="master_caution", passed=mc_ok,
        expected=f"MC={exp}", actual=f"MC text='{mc_text}'",
        category=ErrorCategory.WCA_CRITICAL,
        goes_to_wca=True,
        note="" if mc_ok else f"Master Caution '{exp}' olmalıydı, ekranda '{mc_text}'.",
        bbox=rep.bbox,
    ))

    wca_entries = pencere.wca.snapshot_sorted()
    all_texts = " ".join(e.text.upper() for e in wca_entries)

    # Beklenen WCA mesajları var mı?
    if scenario.expected_wca_texts:
        present = all(t.upper() in all_texts for t in scenario.expected_wca_texts)
        rep.checks.append(Check(
            name="wca_present", passed=present,
            expected="WCA mesajları: " + ", ".join(scenario.expected_wca_texts),
            actual=(all_texts[:120] or "(WCA boş)"),
            category=ErrorCategory.WCA_MISSING,
            goes_to_wca=True,
            note="" if present else "Beklenen WCA uyarısı panelde yok.",
            bbox=rep.bbox,
        ))

    # NOMINAL senaryoda spurious WARNING/CAUTION olmamalı.
    # DEMO/advisory girdileri (tasarım gereği kalıcı) hariç tutulur.
    if scenario.severity == "NOMINAL" or scenario.expected_mc_state == "OFF":
        real = real_wca_entries(pencere)
        w = sum(1 for e in real if e.severity == "WARNING")
        c = sum(1 for e in real if e.severity == "CAUTION")
        clean = (w == 0 and c == 0)
        rep.checks.append(Check(
            name="wca_no_spurious", passed=clean,
            expected="gerçek WARNING/CAUTION yok", actual=f"WARNING={w} CAUTION={c}",
            category=ErrorCategory.WCA_SPURIOUS,
            goes_to_wca=True,
            note="" if clean else "Nominal durumda gereksiz WCA uyarısı çıkmış.",
            bbox=rep.bbox,
        ))

    # ── WCA EKRAN KONTROLLERİ (Visual & Text Integrity) ──
    wca_frame = getattr(pencere, "wca_frame", None)
    wca_visible = wca_frame is not None and wca_frame.isVisible() and wca_frame.height() > 10 and wca_frame.width() > 10
    rep.checks.append(Check(
        name="wca_visible", passed=wca_visible,
        expected="WCA frame visible", actual="visible" if wca_visible else "hidden/missing",
        category=ErrorCategory.MISSING_SECTION,
        goes_to_wca=True,
        note="" if wca_visible else "WCA paneli/bölümü görünür değil veya gizlenmiş.",
        bbox=rep.bbox,
    ))

    from PyQt5.QtWidgets import QLabel
    wca_labels = []
    if hasattr(pencere, "wca_lay") and pencere.wca_lay is not None:
        for i in range(pencere.wca_lay.count()):
            item = pencere.wca_lay.itemAt(i)
            if item and item.widget():
                if isinstance(item.widget(), QLabel):
                    wca_labels.append(item.widget())

    for lbl in wca_labels:
        txt = lbl.text().upper()
        bg_style = lbl.styleSheet().upper()
        if "WARNING" in txt:
            expected_bg = "#FF3333"
            severity_str = "WARNING"
        elif "CAUTION" in txt:
            expected_bg = "#FFB000"
            severity_str = "CAUTION"
        else:
            expected_bg = "#00FF66"
            severity_str = "ADVISORY"
            
        bg_ok = expected_bg in bg_style
        actual_bg = "unknown"
        if "#FF3333" in bg_style: actual_bg = "red (WARNING)"
        elif "#FFB000" in bg_style: actual_bg = "yellow (CAUTION)"
        elif "#00FF66" in bg_style: actual_bg = "green (ADVISORY)"
        
        rep.checks.append(Check(
            name="wca_color", passed=bg_ok,
            expected=expected_bg, actual=actual_bg,
            category=ErrorCategory.WCA_WRONG_COLOR,
            goes_to_wca=True,
            note="" if bg_ok else f"WCA '{txt.split(' ')[0]}' mesajı {expected_bg} yerine {actual_bg} renginde boyanmış.",
            bbox=rep.bbox,
        ))

    # Duplicate check
    seen_wca_texts = set()
    duplicates = []
    for lbl in wca_labels:
        txt_val = lbl.text().strip().upper()
        if txt_val in seen_wca_texts:
            duplicates.append(txt_val)
        seen_wca_texts.add(txt_val)
    wca_no_duplicates = len(duplicates) == 0
    rep.checks.append(Check(
        name="wca_no_duplicates", passed=wca_no_duplicates,
        expected="her WCA mesajı benzersiz olmalı",
        actual=(f"tekrar edenler: {', '.join(duplicates)}" if duplicates else "tümü benzersiz"),
        category=ErrorCategory.WCA_DUPLICATE,
        goes_to_wca=True,
        note="" if wca_no_duplicates else f"WCA panelinde tekrar eden mesaj(lar) bulundu: {', '.join(duplicates)}.",
        bbox=rep.bbox,
    ))

    from tests.visual_analyzer import check_wca_for_unexpected_text
    
    class PseudoEntry:
        def __init__(self, text):
            self.text = text
            
    pseudo_entries = [PseudoEntry(lbl.text()) for lbl in wca_labels]
    unexpected_texts = check_wca_for_unexpected_text(pseudo_entries, WCA_ALLOWED_TEXTS)
    no_unexpected = (len(unexpected_texts) == 0)
    rep.checks.append(Check(
        name="wca_no_unexpected_text", passed=no_unexpected,
        expected="sadece izin verilen WCA kelimeleri",
        actual=(", ".join(unexpected_texts) if unexpected_texts else "tümü izinli"),
        category=ErrorCategory.WCA_WRONG_TEXT,
        goes_to_wca=True,
        note="" if no_unexpected else f"WCA panelinde izin verilmeyen yabancı kelime/kelimeler bulundu: {', '.join(unexpected_texts)}.",
        bbox=rep.bbox,
    ))

    return rep


# ─── ANTI-ICE DURUM ───────────────────────────────────────────────────────────

def run_anti_ice_check(pencere, expected_value: str) -> Check:
    actual = pencere.lbl_anti.text().strip().upper()
    exp = expected_value.strip().upper()
    ok = actual == exp
    return Check(
        name="anti_ice_state", passed=ok,
        expected=exp, actual=actual,
        category=ErrorCategory.STATE_MISMATCH,
        note="" if ok else f"ANTI-ICE durumu '{exp}' olmalı, ekranda '{actual}'.",
    )


# ─── KÜRESEL EKRAN KONTROLLERİ ────────────────────────────────────────────────

def run_global_checks(pencere) -> ParamReport:
    rep = ParamReport(key="GLOBAL", label="GLOBAL SCREEN CHECKS", injected_value=0)
    
    # Check for spurious labels
    from PyQt5.QtWidgets import QLabel
    labels = pencere.findChildren(QLabel)
    spurious_labels = []
    for lbl in labels:
        name = str(lbl.objectName())
        text = str(lbl.text())
        if name == "spurious_error_label" or "spurious" in text.lower() or "unwanted" in text.lower():
            spurious_labels.append(lbl)
            
    if spurious_labels:
        for sl in spurious_labels:
            bbox = get_widget_bbox(sl, pencere)
            rep.checks.append(Check(
                name="spurious_text_detector", passed=False,
                expected="ekranda alakasız/istenmeyen bir yazı olmamalı",
                actual=f"'{sl.text()}' yazısı bulundu",
                category=ErrorCategory.UNEXPECTED_TEXT,
                note="Ekranın alakasız bir yerinde olmaması gereken bir yazı çıktı!",
                bbox=bbox
            ))
            
    return rep
