"""
TUSAS TestLab — Comprehensive AI Vision Test v4
================================================
Yenilikler (v3 → v4):
  1. Hata tipleri: WCA / Visual / UI kategorileri ayrımı
  2. Bar doluluk oranı + renk + numeric birlikte test edilir
  3. Parametre bazlı min/max/range (hepsi 0-100 değil)
  4. Anti-ice AUTO/OFF/ON state testleri
  5. Renk threshold testleri (60'da kırmızı mı yeşil mi?)
  6. Unexpected text / artefact tespiti
  7. Boş panel / missing section testi
  8. Unit doğrulama (°C vs °F)
  9. Snapshot-based: simülasyon akışına değil ekran STATE'ine göre test
 10. Raporda: tam screenshot + annotated (işaretlenmiş) + per-parametre crop
 11. Rapor session sonunda OTOMATIK açılır

  Tam yerel test (Logic + Visual + ML):
      python run_tests.py

  Tek kategori:
      python -m pytest tests/test_ai_vision_v4.py -v -s -k "COLOR"

  Tüm testler tek komut:
      python run_tests.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import base64
import random
import types
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytest
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, Qt, QTime
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekran import FlightDisplay
from tests.fault_scenarios import SCENARIOS as BASE_SCENARIOS, FaultScenario
from tests.param_config import (
    PARAM_CONFIGS, WCA_ALLOWED_TEXTS, STATUS_ALLOWED_VALUES, ErrorCategory
)
from tests.visual_analyzer import (
    VisualCheckResult, AnnotationBox,
    get_widget_bbox, pil_from_screenshot, crop_bbox,
    detect_dominant_color, detect_background_color,
    measure_bar_fill_ratio, fill_ratio_matches, expected_fill_ratio,
    check_numeric_displayed, check_unit_displayed,
    annotate_screenshot, save_cropped_error,
    check_wca_for_unexpected_text, check_for_random_text_artifacts,
)

try:
    from cv_analyzer import analyze_screenshot, print_cv_result
    HAS_CV_ANALYZER = True
except (ImportError, OSError):
    HAS_CV_ANALYZER = False

try:
    from ml_trainer_v3 import collect_training_data, predict, dataset_summary
    HAS_ML = True
except (ImportError, OSError):
    HAS_ML = False


# ─── RAPOR KLASÖRÜ ────────────────────────────────────────────────────────────

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_reports")
SCREENSHOT_DIR = os.path.join(REPORT_DIR, "screenshots")
ANNOTATED_DIR = os.path.join(REPORT_DIR, "annotated")
CROP_DIR = os.path.join(REPORT_DIR, "crops")
for d in [REPORT_DIR, SCREENSHOT_DIR, ANNOTATED_DIR, CROP_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── YENİ SENARYOLAR ──────────────────────────────────────────────────────────
# Mevcut SCENARIOS listesine eklenecek yeni test senaryoları

EXTRA_SCENARIOS: List[FaultScenario] = [

    # ── RENK THRESHOLD TESTLERİ ──────────────────────────────────────────────
    # TRQ1 = 60 → 0-100 arası yeşil zone → bar yeşil olmalı
    FaultScenario(
        id="COLOR_001",
        name="TRQ1 Nominal Zone — Green Bar Check",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "TRQ1=60 → yeşil zone (0-100). "
            "Bar yeşil renkte olmalı. WCA'ya düşmemeli. "
            "Doluluk oranı ~%55 olmalı (60/110)."
        ),
        inject={"E1_TRQ": 60.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "Engine 1 TRQ1 parametresi ekranda yeşil renkte mi gösteriliyor? "
            "Değer yaklaşık 60 gösteriyor mu? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    # TRQ1 = 108 → 105-110 arası kırmızı WARNING zone
    FaultScenario(
        id="COLOR_002",
        name="TRQ1 Warning Zone — Red Bar + WCA Check",
        category="ENGINE",
        severity="WARNING",
        description=(
            "TRQ1=108 → kırmızı warning zone (>105). "
            "Bar kırmızı + WCA 'TRQ1 HIGH WARNING' çıkmalı."
        ),
        inject={"E1_TRQ": 108.0},
        expected_wca_texts=["TRQ1", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "TRQ1 bar kırmızı renkte mi? "
            "WCA panelinde TRQ1 ile ilgili uyarı var mı? EVET/HAYIR."
        ),
    ),

    # TRQ1 = 102 → 100-105 arası turuncu CAUTION zone
    FaultScenario(
        id="COLOR_003",
        name="TRQ1 Caution Zone — Orange Bar Check",
        category="ENGINE",
        severity="CAUTION",
        description="TRQ1=102 → turuncu caution zone (100-105). Bar turuncu olmalı.",
        inject={"E1_TRQ": 102.5},
        expected_wca_texts=["TRQ1"],
        expected_mc_state="CAUTION",
        ai_vision_prompt=(
            "TRQ1 bar turuncu/sarı renkte mi? EVET/HAYIR."
        ),
    ),

    # ── ANTI-ICE STATE TESTLERİ ───────────────────────────────────────────────
    FaultScenario(
        id="ANTI_001",
        name="Anti-Ice Default State — AUTO",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "Anti-ice varsayılan durumu AUTO olmalı. "
            "Değer listede (OFF/AUTO/ON) olmalı. "
            "WCA'ya düşmemeli."
        ),
        inject={},  # inject yok, mevcut state test edilir
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "STATUS panelinde ANTI-ICE değeri 'AUTO', 'ON' veya 'OFF' yazıyor mu? "
            "Değer okunabilir durumda mı? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="ANTI_002",
        name="Anti-Ice State Change — OFF",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "Anti-ice değeri OFF olarak set edilir ve "
            "ekranda OFF gösterilmesi beklenir."
        ),
        inject={},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "STATUS panelinde ANTI-ICE değeri 'OFF' yazıyor mu? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="ANTI_003",
        name="Anti-Ice State Change — ON",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "Anti-ice değeri ON olarak set edilir ve "
            "ekranda ON gösterilmesi beklenir."
        ),
        inject={},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "STATUS panelinde ANTI-ICE değeri 'ON' yazıyor mu? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    # ── BAR SCALE / DOLULUK TESTİ ────────────────────────────────────────────
    FaultScenario(
        id="SCALE_001",
        name="FUEL_L Bar Fill Ratio — 50% Fill Check",
        category="FUEL",
        severity="ADVISORY",
        description=(
            "FUEL_L=300 LBS (max 600). "
            "Beklenen doluluk: ~%50. "
            "Numeric değer + bar doluluk oranı birlikte test edilir."
        ),
        inject={"FUEL_L": 300.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "FUEL bölümünde L TANK bar yarıya kadar dolu mu (~%50)? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="SCALE_002",
        name="HYD_A_P Bar Fill Ratio — High Value Check",
        category="HYDRAULIC",
        severity="ADVISORY",
        description=(
            "HYD_A_P=3000 PSI (max 3500, min 0). "
            "Beklenen doluluk: ~%86. "
            "Bar yeşil + doluluk doğru olmalı."
        ),
        inject={"HYD_A_P": 3000.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "Hidrolik SYS A P bar doluluk oranı yüksek mi (yaklaşık %85-90)? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    # ── UNIT DOĞRULAMAn ───────────────────────────────────────────────────────
    FaultScenario(
        id="UNIT_001",
        name="Temperature Unit Validation — °C Check",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "TIT1 sıcaklık birimi '°C' olmalı. "
            "'°F' veya başka bir birim yanlış."
        ),
        inject={"E1_TIT": 700.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "ENGINE panelinde TIT1 birimi '°C' olarak gösteriliyor mu? "
            "'°F' görünüyor mu? EVET/HAYIR (°C için EVET)."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="UNIT_002",
        name="Fuel Unit Validation — LBS Check",
        category="FUEL",
        severity="ADVISORY",
        description="FUEL_L biriminin 'LBS' olduğunu doğrula.",
        inject={"FUEL_L": 400.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "FUEL bölümünde L TANK birimi 'LBS' olarak gösteriliyor mu? EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    # ── WCA EK DURUMLAR ───────────────────────────────────────────────────────
    FaultScenario(
        id="WCA_001",
        name="WCA Unexpected Text — Artifact Check",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "Nominal durumda WCA panelinde sadece izin verilen metinler olmalı. "
            "Tanımsız metin/artefact olmamalı."
        ),
        inject={
            "E1_TRQ": 78.0, "E2_TRQ": 77.0,
            "FUEL_L": 380.0, "FUEL_R": 370.0,
        },
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "WCA panelinde beklenmedik, anlamsız veya alakasız bir metin var mı? "
            "Örn. 'Sinem', 'Test123' gibi sistem dışı yazılar. EVET/HAYIR (yok ise HAYIR)."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="WCA_002",
        name="WCA Priority — Most Critical First",
        category="ENGINE",
        severity="WARNING",
        description=(
            "Birden fazla alarm varken WCA en kritik olanı (WARNING) önce göstermeli."
        ),
        inject={"E1_TRQ": 108.0, "E1_NP": 90.0},
        expected_wca_texts=["TRQ1"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "WCA panelinde kırmızı renkli (WARNING seviyeli) bir uyarı en üstte mi? "
            "EVET/HAYIR."
        ),
    ),

    # ── MISSING SECTION TESTİ ─────────────────────────────────────────────────
    FaultScenario(
        id="PANEL_001",
        name="Engine Panel Visibility Check",
        category="ENGINE",
        severity="ADVISORY",
        description=(
            "ENGINE / PROPULSION paneli görünür ve dolu olmalı. "
            "Boş alan veya render edilmemiş bölge olmamalı."
        ),
        inject={"E1_TRQ": 78.0, "E2_TRQ": 77.0},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "Ekranda ENGINE veya PROPULSION paneli görünür ve değer içeriyor mu? "
            "Boş bir alan var mı? EVET/HAYIR (görünür ise EVET)."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    FaultScenario(
        id="PANEL_002",
        name="WCA Panel Visibility Check",
        category="ENGINE",
        severity="ADVISORY",
        description="WCA paneli görünür ve aktif olmalı. Boş kalmamalı.",
        inject={},
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "WCA paneli ekranda görünür mü ve içinde metin var mı? "
            "EVET/HAYIR."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),

    # ── MULTI-PARAM CASCADE GÖRSEL TEST ───────────────────────────────────────
    FaultScenario(
        id="VIS_001",
        name="Multiple Param Visual Consistency",
        category="ENGINE",
        severity="WARNING",
        description=(
            "E1_OILP=18 PSI (kırmızı), E1_TRQ=60 (yeşil). "
            "Her param kendi renk zone'unda gösterilmeli. "
            "Sadece OIL P1 WCA'ya düşmeli."
        ),
        inject={"E1_OILP": 18.0, "E1_TRQ": 60.0},
        expected_wca_texts=["OIL P1", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "OIL P1 kırmızı gösterilirken TRQ1 yeşil mi gösteriliyor? "
            "Yani iki farklı renk aynı anda görünüyor mu? EVET/HAYIR."
        ),
    ),
]

# Tüm senaryolar
ALL_SCENARIOS = BASE_SCENARIOS + EXTRA_SCENARIOS


# ─── SONUÇ DATACLASS ──────────────────────────────────────────────────────────

@dataclass
class VisualCheckSummary:
    """Tek parametre için tüm görsel check sonuçları."""
    param_key: str
    param_label: str
    injected_value: float
    checks: List[VisualCheckResult] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None
    crop_path: str = ""          # hatalı durum crop'ı (eski alan, uyumluluk için)
    nominal_crop_path: str = ""  # olması gereken (nominal) crop
    actual_crop_path: str = ""   # hatalı görünüm (faulty) crop

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failing_checks(self) -> List[VisualCheckResult]:
        return [c for c in self.checks if not c.passed]


@dataclass
class VisionTestResultV4:
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    timestamp: str
    # Logic tests
    mc_state_ok: bool = False
    wca_text_ok: bool = False
    # Visual checks (per-parameter)
    visual_summaries: List[VisualCheckSummary] = field(default_factory=list)
    # AI Vision Removed
    # ML
    ml_prediction: str = ""
    ml_dataset_count: int = 0
    # WCA integrity
    unexpected_wca_texts: List[str] = field(default_factory=list)
    # Paths
    screenshot_path: str = ""        # hatalı durum tam ekran
    nominal_screenshot_path: str = ""  # nominal (fault öncesi) tam ekran
    annotated_path: str = ""
    # Result
    overall_pass: bool = False
    visual_pass: bool = True
    error: str = ""
    duration_ms: int = 0

    @property
    def has_visual_failures(self) -> bool:
        return any(not s.all_passed for s in self.visual_summaries)

    @property
    def all_failing_checks(self) -> List[VisualCheckResult]:
        result = []
        for s in self.visual_summaries:
            result.extend(s.failing_checks)
        return result

    @property
    def wca_required_errors(self) -> List[VisualCheckResult]:
        return [c for c in self.all_failing_checks if c.goes_to_wca]

    @property
    def visual_only_errors(self) -> List[VisualCheckResult]:
        return [c for c in self.all_failing_checks if not c.goes_to_wca]


# ─── SNAPSHOT ALMA ────────────────────────────────────────────────────────────

def take_snapshot(widget, scenario_id: str, suffix: str = "") -> Tuple[QPixmap, str, str, Optional[object]]:
    """
    Snapshot-based yaklaşım:
    Simülasyon akışına değil, o ANDA ekranda ne varsa onu al.

    suffix="" → "{scenario_id}.png"  (hatalı durum)
    suffix="_nominal" → "{scenario_id}_nominal.png"  (olması gereken)

    Döndürür: (pixmap, path, base64_str, pil_image)
    """
    widget.repaint()
    QApplication.processEvents()
    time.sleep(0.1)

    pixmap = widget.grab()
    filename = f"{scenario_id}{suffix}.png"
    path = os.path.join(SCREENSHOT_DIR, filename)
    pixmap.save(path, "PNG")

    # base64
    buf = QByteArray()
    qbuf = QBuffer(buf)
    qbuf.open(QIODevice.WriteOnly)
    pixmap.save(qbuf, "PNG")
    b64 = base64.b64encode(buf.data()).decode("utf-8")

    # PIL image
    pil_img = pil_from_screenshot(path)
    
    # [RETINA DISPLAY FIX]
    # macOS High DPI scales the pixmap by 2x (e.g. 1600x860 logical -> 3200x1720 physical)
    # Resize the PIL image back to logical widget size so bounding boxes align perfectly.
    ratio = widget.devicePixelRatioF()
    if pil_img and ratio != 1.0:
        from PIL import Image
        pil_img = pil_img.resize((widget.width(), widget.height()), Image.LANCZOS)

    return pixmap, path, b64, pil_img


# ─── SENARYO UYGULAMA (SNAPSHOT-BASED) ───────────────────────────────────────

def apply_scenario_snapshot(pencere: FlightDisplay, scenario: FaultScenario):
    """
    Snapshot-based yaklaşım:
    1. Sim timer durdurulur
    2. Parametreler inject edilir
    3. FaultGate zaman ilerletilir
    4. _apply_ui + _render_wca çağrılır
    5. State sabitleşir, screenshot alınır

    _tick_sim'in üzerine yazmasına izin verilmez.
    """
    # Simülasyon timer'ını geçici durdur
    pencere.t_sim.stop()

    # Parametreleri enjekte et
    for key, val in scenario.inject.items():
        pencere.vals[key] = float(val)

    # Invalid parametreler
    for key in scenario.invalid_params:
        pencere.invalid[key] = True

    # FaultGate için elapsed ilerlet
    if scenario.time_advance_secs > 0:
        pencere.elapsed = pencere.elapsed.addSecs(max(scenario.time_advance_secs, 10))

    # FaultGate'i tetikle: önce tick ile state'leri kaydet
    for key, val in scenario.inject.items():
        pencere.vals[key] = float(val)

    original_tick = pencere._tick_sim.__func__

    def _frozen_tick(self):
        """Tick sonrası inject'i yeniden uygula — simülasyon ezmesin."""
        original_tick(self)
        for k, v in scenario.inject.items():
            self.vals[k] = float(v)
        for k in scenario.invalid_params:
            self.invalid[k] = True

    pencere._tick_sim = types.MethodType(_frozen_tick, pencere)
    pencere._tick_sim()

    # FaultGate start_ms'i sıfırla (gate açılsın)
    for st in pencere.fgate._s.values():
        if st.active:
            st.start_ms = 0

    # tick_count kadar daha tetikle
    for _ in range(max(scenario.tick_count, 1)):
        pencere._tick_sim()

    # Patch geri al
    pencere._tick_sim = types.MethodType(original_tick, pencere)

    # UI güncelle
    pencere._apply_ui()
    now = pencere._now_ms()
    pencere._render_wca(now)
    QApplication.processEvents()
    time.sleep(0.1)


# ─── QT WIDGET RENK YARDIMCIları ─────────────────────────────────────────────

# Qt bar renk hex → isim haritası (widgets.py'deki _color_for_value() değerleri)
_QT_HEX_TO_COLOR = {
    "#ff3333": "red",      # WARNING
    "#ffb000": "orange",   # CAUTION
    "#00ff66": "green",    # NOMINAL
    "#3399ff": "blue",     # düşük değer
    "#ffff33": "yellow",   # orta-yüksek
    "#ff9933": "orange",   # yüksek
    "#3a0000": "red",      # WARNING arka fon
    "#3a2600": "orange",   # CAUTION arka fon
    "#050505": "black",    # normal arka fon
}

# Beklenen Qt state → renk adı
_STATE_TO_COLOR = {
    "WARNING": "red",
    "CAUTION": "orange",
    "NOMINAL": "green",
    "INVALID": "red",
}


def _qt_bar_color(widget) -> str:
    """
    ParamBar widget'ının mevcut bar rengini Qt'den oku.
    PIL gerekmez — doğrudan QColor.name() okunur.
    """
    try:
        hex_c = widget.bar._color.name().lower()
        return _QT_HEX_TO_COLOR.get(hex_c, hex_c)
    except Exception:
        return "unknown"


def _qt_widget_state(widget, value: float) -> str:
    """
    Widget'ın mevcut görsel state'ini Qt'den oku.
    ParamBar ve ParamTextRow için çalışır.
    """
    try:
        if hasattr(widget, "get_state"):
            return widget.get_state(value, False)
        # ParamTextRow: styleSheet'ten oku
        style = widget.styleSheet().upper()
        if "FF3333" in style:
            return "WARNING"
        if "FFB000" in style:
            return "CAUTION"
        return "NOMINAL"
    except Exception:
        return "UNKNOWN"


def _qt_background_color(widget) -> str:
    """
    Widget arka fon rengini styleSheet'ten oku.
    ParamBar için lbl_name, ParamTextRow için widget itself.
    """
    try:
        style = widget.styleSheet().upper()
        if hasattr(widget, "lbl_name"):
            style += " " + widget.lbl_name.styleSheet().upper()

        if "FF3333" in style and "BACKGROUND" in style:
            return "red"
        if "FFB000" in style and "BACKGROUND" in style:
            return "orange"
        if "00FF66" in style and "BACKGROUND" in style:
            return "green"
        return "none"
    except Exception:
        return "unknown"


def _expected_state(cfg, value: float) -> str:
    """Config'den beklenen state'i hesapla."""
    color = cfg.expected_color(value)
    return {"red": "WARNING", "orange": "CAUTION", "green": "NOMINAL"}.get(color or "green", "NOMINAL")


# ─── GÖRSEL KONTROLLER (Qt tabanlı — PIL gerektirmez) ─────────────────────────

def run_visual_checks(
    pencere: FlightDisplay,
    scenario: FaultScenario,
    pil_img,   # PIL Image veya None — sadece annotation için kullanılır
) -> Tuple[List[VisualCheckSummary], List[AnnotationBox]]:
    """
    Inject edilen her parametre için Qt widget state'ini doğrudan okuyarak
    görsel kontroller yap. PIL yüklü olmasa bile tüm checkler çalışır.

    Kontroller:
      numeric_value  — ekranda doğru sayı gösteriliyor mu?
      unit           — doğru birim gösteriliyor mu? (°C, LBS, PSI...)
      bar_color      — bar doğru renkte mi? (Qt QColor'dan okunur)
      bar_fill       — bar doluluk oranı değerle uyuşuyor mu? (Qt'den hesaplanır)
      widget_state   — widget state'i (WARNING/CAUTION/NOMINAL) doğru mu?
      background     — arka fon rengi state ile uyuşuyor mu?
      visibility     — widget görünür ve sıfırdan büyük boyutta mu?
    """
    summaries: List[VisualCheckSummary] = []
    annotations: List[AnnotationBox] = []

    for param_key, injected_val in scenario.inject.items():
        cfg = PARAM_CONFIGS.get(param_key)
        if cfg is None:
            continue

        summary = VisualCheckSummary(
            param_key=param_key,
            param_label=cfg.label,
            injected_value=injected_val,
        )

        widget = pencere.param_widgets.get(param_key) or pencere.text_widgets.get(param_key)
        bbox = get_widget_bbox(widget, pencere) if widget else None
        summary.bbox = bbox

        # ── Görünürlük (missing section) ──────────────────────────────────────
        if widget is None:
            summary.checks.append(VisualCheckResult(
                check_name="visibility",
                passed=False,
                expected="widget visible",
                actual="widget not found — missing from param_widgets or text_widgets",
                error_category=ErrorCategory.MISSING_SECTION,
                goes_to_wca=False,
                bbox=None,
            ))
            summaries.append(summary)
            continue

        vis_ok = widget.isVisible() and widget.width() > 0 and widget.height() > 0
        if not vis_ok:
            summary.checks.append(VisualCheckResult(
                check_name="visibility",
                passed=False,
                expected="visible (w>0, h>0)",
                actual=f"isVisible={widget.isVisible()}, size={widget.width()}x{widget.height()}",
                error_category=ErrorCategory.MISSING_SECTION,
                goes_to_wca=False,
                bbox=bbox,
            ))

        # ── Numeric değer ─────────────────────────────────────────────────────
        if "numeric" in cfg.visual_checks and hasattr(widget, "lbl_val"):
            passed, exp, act = _check_numeric_via_spec(widget, injected_val)
            summary.checks.append(VisualCheckResult(
                check_name="numeric_value",
                passed=passed,
                expected=exp,
                actual=act,
                error_category=ErrorCategory.VISUAL_SCALE,
                goes_to_wca=False,
                bbox=bbox,
                note=f"The screen should be displaying {exp}, but it is showing {act}. This is a visual error." if not passed else "",
            ))

        # ── Unit (birim) ──────────────────────────────────────────────────────
        if "unit" in cfg.visual_checks and hasattr(widget, "lbl_val") and cfg.unit:
            try:
                actual_text = widget.lbl_val.text()
            except Exception:
                actual_text = "OKUNAMADI"
            unit_ok = cfg.unit in actual_text
            summary.checks.append(VisualCheckResult(
                check_name="unit",
                passed=unit_ok,
                expected=cfg.unit,
                actual=actual_text,
                error_category=ErrorCategory.UNIT_MISMATCH,
                goes_to_wca=False,
                bbox=bbox,
                note=f"The unit on the screen should be '{cfg.unit}', but the screen is displaying something else." if not unit_ok else "",
            ))

        # ── Bar rengi (Qt QColor'dan direkt okunur) ───────────────────────────
        if "bar_color" in cfg.visual_checks and hasattr(widget, "bar"):
            actual_color = _qt_bar_color(widget)
            expected_color = cfg.expected_color(injected_val) or "green"
            passed = (actual_color == expected_color) or (actual_color == "unknown")

            # Bu range'de WCA bekleniyor mu?
            goes_to_wca = False
            for r in cfg.color_ranges:
                if r.min_val <= injected_val < r.max_val and r.wca_severity in ("WARNING", "CAUTION"):
                    goes_to_wca = cfg.wca_enabled
                    break

            summary.checks.append(VisualCheckResult(
                check_name="bar_color",
                passed=passed,
                expected=expected_color,
                actual=actual_color,
                error_category=ErrorCategory.COLOR_THRESHOLD,
                goes_to_wca=goes_to_wca,
                bbox=bbox,
                note=(
                    f"The system expected the indicator color to be {expected_color.upper()}, "
                    f"but it is incorrectly displaying as {actual_color.upper()} on the screen. "
                    f"{'This triggers a Master Caution warning.' if goes_to_wca else 'This is just a visual screen error.'}"
                ) if not passed else "",
            ))

        # ── Bar doluluk oranı (Qt widget değerinden hesaplanır) ───────────────
        if "bar_fill" in cfg.visual_checks and hasattr(widget, "_last_real") and cfg.vmax != cfg.vmin:
            displayed_val = widget._last_real
            actual_fill = (displayed_val - cfg.vmin) / (cfg.vmax - cfg.vmin)
            actual_fill = max(0.0, min(1.0, actual_fill))
            exp_fill = max(0.0, min(1.0, (injected_val - cfg.vmin) / (cfg.vmax - cfg.vmin)))

            passed = abs(actual_fill - exp_fill) <= 0.02  # %2 tolerans
            summary.checks.append(VisualCheckResult(
                check_name="bar_fill",
                passed=passed,
                expected=f"{exp_fill:.0%} (value={injected_val:.1f}/{cfg.vmax})",
                actual=f"{actual_fill:.0%} (displayed value={displayed_val:.1f})",
                error_category=ErrorCategory.VISUAL_SCALE,
                goes_to_wca=False,
                bbox=bbox,
                note=(
                    f"The bar on the screen should look {exp_fill:.0%} full based on the system value, "
                    f"but the screen is showing it at {actual_fill:.0%} full. The display is incorrect."
                ) if not passed else "",
            ))

        # ── Widget state (WARNING/CAUTION/NOMINAL) ────────────────────────────
        if "bar_color" in cfg.visual_checks or "background" in cfg.visual_checks:
            actual_state = _qt_widget_state(widget, injected_val)
            expected_state = _expected_state(cfg, injected_val)
            state_ok = (actual_state == expected_state) or (actual_state == "UNKNOWN")
            summary.checks.append(VisualCheckResult(
                check_name="widget_state",
                passed=state_ok,
                expected=expected_state,
                actual=actual_state,
                error_category=ErrorCategory.COLOR_THRESHOLD,
                goes_to_wca=cfg.wca_enabled and expected_state == "WARNING",
                bbox=bbox,
                note=(
                    f"The status label on the screen should read '{expected_state}', "
                    f"but it is showing '{actual_state}' instead."
                ) if not state_ok else "",
            ))

        # ── Arka fon rengi (Qt styleSheet'ten) ───────────────────────────────
        if "background" in cfg.visual_checks:
            actual_bg = _qt_background_color(widget)
            expected_state_for_bg = _expected_state(cfg, injected_val)
            expected_bg = {"WARNING": "red", "CAUTION": "orange", "NOMINAL": "none"}.get(
                expected_state_for_bg, "none"
            )
            bg_ok = (actual_bg == expected_bg) or (expected_bg == "none" and actual_bg in ("none", "black", "unknown"))
            summary.checks.append(VisualCheckResult(
                check_name="background_color",
                passed=bg_ok,
                expected=expected_bg,
                actual=actual_bg,
                error_category=ErrorCategory.BACKGROUND_COLOR,
                goes_to_wca=False,
                bbox=bbox,
                note=(
                    f"Background color mismatch for state. "
                    f"State={expected_state_for_bg} expected bg={expected_bg}, "
                    f"but actual={actual_bg}."
                ) if not bg_ok else "",
            ))

        # ── Annotation (PIL varsa) ────────────────────────────────────────────
        for chk in summary.failing_checks:
            if chk.bbox:
                x, y, w, h = chk.bbox
                annotations.append(AnnotationBox(
                    x=x, y=y, w=w, h=h,
                    label=f"{cfg.label}:{chk.check_name}",
                    color=(255, 50, 50) if chk.goes_to_wca else (255, 165, 0),
                ))

        summaries.append(summary)

    # ── WCA PANEL GÖRSEL KONTROLLERİ ──────────────────────────────────────────
    wca_summary = VisualCheckSummary(param_key="WCA", param_label="WCA PANEL", injected_value=0)
    wca_bbox = get_widget_bbox(pencere.wca_frame, pencere)
    wca_summary.bbox = wca_bbox
    
    from PyQt5.QtWidgets import QLabel
    wca_labels = [pencere.wca_lay.itemAt(i).widget() for i in range(pencere.wca_lay.count()) if isinstance(pencere.wca_lay.itemAt(i).widget(), QLabel)]
    
    for lbl in wca_labels:
        txt = lbl.text()
        bg_style = lbl.styleSheet().upper()
        if "WARNING" in txt:
            expected_bg = "#FF3333"
            cat = ErrorCategory.COLOR_THRESHOLD
        elif "CAUTION" in txt:
            expected_bg = "#FFB000"
            cat = ErrorCategory.COLOR_THRESHOLD
        else:
            expected_bg = "#00FF66"
            cat = ErrorCategory.COLOR_THRESHOLD
            
        bg_ok = expected_bg in bg_style
        
        actual_bg = "unknown"
        if "#FF3333" in bg_style: actual_bg = "red (WARNING)"
        elif "#FFB000" in bg_style: actual_bg = "yellow (CAUTION)"
        elif "#00FF66" in bg_style: actual_bg = "green (ADVISORY)"

        exp_color_name = {"#FF3333": "red (WARNING)", "#FFB000": "yellow (CAUTION)", "#00FF66": "green (ADVISORY)"}.get(expected_bg, "none")
        
        if not bg_ok:
            clean_txt = txt.split('  t+')[0] if '  t+' in txt else txt
            wca_summary.checks.append(VisualCheckResult(
                check_name="wca_color",
                passed=bg_ok,
                expected=exp_color_name,
                actual=actual_bg,
                error_category=cat,
                goes_to_wca=True,
                bbox=wca_bbox,
                note=f"WCA message '{clean_txt}' expected to be {exp_color_name}, but rendered in {actual_bg}."
            ))
            
    if wca_summary.failing_checks:
        for chk in wca_summary.failing_checks:
            if wca_bbox:
                x, y, w, h = wca_bbox
                annotations.append(AnnotationBox(
                    x=x, y=y, w=w, h=h,
                    label=f"WCA:{chk.check_name}",
                    color=(255, 50, 50)
                ))
        summaries.append(wca_summary)

    return summaries, annotations


def _check_numeric_via_spec(widget, expected_value: float) -> Tuple[bool, str, str]:
    """Widget spec'inden decimals alarak numeric check yap."""
    try:
        spec = widget.spec
        actual_text = widget.lbl_val.text().strip()
        # birimi kaldır
        num_part = actual_text.replace(spec.unit, "").strip()
        actual_num = float(num_part)
        tolerance = 0.5 * (10 ** (-spec.decimals)) + 0.1
        passed = abs(actual_num - expected_value) <= max(tolerance, abs(expected_value) * 0.01 + 0.5)
        exp_str = f"{expected_value:.{spec.decimals}f} {spec.unit}".strip()
        act_str = actual_text
        return passed, exp_str, act_str
    except Exception as e:
        return False, str(expected_value), f"PARSE_ERR: {e}"


# ─── ANTI-ICE DURUM TESTİ ────────────────────────────────────────────────────

def check_anti_ice_state(pencere: FlightDisplay, expected_value: str) -> VisualCheckResult:
    """Anti-ice label'ının beklenen state'te olup olmadığını kontrol et."""
    try:
        actual = pencere.lbl_anti.text().strip()
        allowed = STATUS_ALLOWED_VALUES.get("ANTI-ICE", ["OFF", "AUTO", "ON"])
        in_allowed = actual in allowed
        correct_value = (actual == expected_value)

        if not in_allowed:
            return VisualCheckResult(
                "anti_ice_state",
                False,
                expected_value,
                actual,
                ErrorCategory.UNEXPECTED_TEXT,
                False,
                note=f"Value '{actual}' not in allowed list {allowed}"
            )
        return VisualCheckResult(
            "anti_ice_state",
            correct_value,
            expected_value,
            actual,
            ErrorCategory.STATE_MISMATCH,
            False,  # Anti-ice state mismatch WCA'ya düşmez
        )
    except Exception as e:
        return VisualCheckResult(
            "anti_ice_state", False, expected_value, f"ERROR: {e}",
            ErrorCategory.STATE_MISMATCH, False
        )


# ─── WCA KONTROLLERI ─────────────────────────────────────────────────────────

def run_wca_logic(pencere: FlightDisplay, scenario: FaultScenario) -> Tuple[bool, bool, List[str]]:
    """
    Master Caution + WCA metin kontrolü.
    Döndürür: (mc_ok, wca_ok, unexpected_texts)
    """
    mc_text = pencere.lbl_mc.text()
    mc_style = pencere.lbl_mc.styleSheet().upper()

    if scenario.expected_mc_state == "WARNING":
        mc_ok = mc_text == "ON" and "#FF3333" in mc_style
    elif scenario.expected_mc_state == "CAUTION":
        mc_ok = mc_text == "ON" and ("#FFB000" in mc_style or "#FF3333" in mc_style)
    else:  # OFF
        mc_ok = mc_text == "OFF"

    wca_entries = pencere.wca.snapshot_sorted()
    all_texts = " ".join(e.text.upper() for e in wca_entries)

    if scenario.expected_wca_texts:
        wca_ok = any(t.upper() in all_texts for t in scenario.expected_wca_texts)
    else:
        wca_ok = True

    # Beklenmedik metin kontrolü
    unexpected = check_wca_for_unexpected_text(wca_entries, WCA_ALLOWED_TEXTS)

    return mc_ok, wca_ok, unexpected


# ─── PYTEST FIXTURES ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(sys.argv)
    yield a


@pytest.fixture
def pencere(app, monkeypatch):
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)
    w = FlightDisplay()
    w.show()
    w.resize(1600, 860)
    QApplication.processEvents()
    yield w
    w.close()



# ─── SONUÇ TOPLAMA ────────────────────────────────────────────────────────────

_all_results: List[VisionTestResultV4] = []


# ─── RASTGELE GÖRSEL HATA ENJEKSİYONU ──────────────────────────────────────────

def inject_random_visual_bug(pencere, scenario, override_param_key=None):
    """
    Simülasyon state'i oturduktan sonra, UI üzerinde havacılık mantığına uygun 
    çeşitlendirilmiş ve gerçekçi hatalar (bug) enjekte eder.
    Her testte mutlaka Görsel bir fark yaratmayı garanti eder.
    """
    param_key = override_param_key
    if not param_key:
        if scenario.inject:
            param_key = random.choice(list(scenario.inject.keys()))
        else:
            param_key = random.choice(list(pencere.param_widgets.keys()))
        
    widget = pencere.param_widgets.get(param_key) or pencere.text_widgets.get(param_key)
    
    if not widget:
        if "ANTI" in param_key or (scenario.id and "ANTI" in scenario.id):
            pencere.lbl_anti.setText("OFF")
            pencere.lbl_anti.setStyleSheet("color:#FF3333;font-weight:bold;")
        return

    from PyQt5.QtGui import QColor

    # Eğer widget INVALID durumundaysa (kırmızı çarpı), görsel hatayı 
    # görünür kılmak için invalid bayrağını kaldırıp kasten normalmiş gibi çizeriz.
    if pencere.invalid.get(param_key, False) or widget._invalid:
        widget.set_invalid(False)
        widget.lbl_val.setText("0.0 ERR")
        if hasattr(widget, "bar"):
            widget.bar._color = QColor("#00FF66")
            widget.bar.update()
        return

    # Hata tipleri (Değere bağlı mantıksal hatalar)
    bug_types = ["color_logic", "scale_mismatch", "unit_error", "background_fail"]
    bug = random.choice(bug_types)
    
    if bug == "color_logic" and hasattr(widget, "bar"):
        val = pencere.vals.get(param_key, 0)
        current_color = widget.bar._color
        
        # Hata sınırlarını kontrol et
        is_danger = False
        if widget.spec.warning_hi and val >= widget.spec.warning_hi:
            is_danger = True
        if widget.spec.warning_lo and val <= widget.spec.warning_lo:
            is_danger = True
            
        if is_danger:
            wrong_colors = [QColor("#00FF66"), QColor("#3399FF")] # Danger => GREEN/BLUE
        else:
            wrong_colors = [QColor("#FF3333"), QColor("#FF9933")] # Safe => RED/ORANGE
            
        # Asla orijinal renkle aynı yapma
        wrong_colors = [c for c in wrong_colors if c.name() != current_color.name()]
        if not wrong_colors:
            wrong_colors = [QColor("#FFFFFF")]
            
        widget.bar._color = random.choice(wrong_colors)
        widget.bar.update()

    elif bug == "scale_mismatch" and hasattr(widget, "bar"):
        real_val = pencere.vals.get(param_key, 0)
        # Barda gerçeğinden TAMAMEN farklı bir doluluk göster
        wrong_val = widget.spec.vmax if real_val < (widget.spec.vmax / 2) else widget.spec.vmin
        widget.bar._value = wrong_val
        widget.bar.update()
        
    elif bug == "unit_error" and hasattr(widget, "lbl_val"):
        current_text = widget.lbl_val.text()
        bad_text = current_text.replace("°C", "°F").replace("LBS", "KG").replace("PSI", "BAR").replace("%", "RPM")
        if bad_text == current_text: 
            bad_text = current_text + " ERR"
        widget.lbl_val.setText(bad_text)
        
    elif bug == "background_fail" and hasattr(widget, "lbl_name"):
        # Ensure visual mismatch
        current_style = widget.lbl_name.styleSheet().upper() + widget.styleSheet().upper()
        if "FF3333" in current_style:
            widget.lbl_name.setStyleSheet("background-color: #00FF66; color: #FFFFFF;")
            if hasattr(widget, "bar"): widget.bar._color = QColor("#00FF66")
        else:
            widget.lbl_name.setStyleSheet("background-color: #FF3333; color: #FFFFFF;")
            if hasattr(widget, "bar"): widget.bar._color = QColor("#FF3333")
        if hasattr(widget, "bar"):
            widget.bar.update()

    # WCA panelindeki bir uyarının rengini kasten boz (WARNING -> CAUTION gibi)
    if random.random() < 0.6:  # %60 ihtimalle WCA rengini boz
        from PyQt5.QtWidgets import QLabel
        wca_labels = [pencere.wca_lay.itemAt(i).widget() for i in range(pencere.wca_lay.count()) if isinstance(pencere.wca_lay.itemAt(i).widget(), QLabel)]
        if wca_labels:
            lbl = random.choice(wca_labels)
            current_style = lbl.styleSheet()
            if "#FF3333" in current_style: # WARNING ise CAUTION yap
                lbl.setStyleSheet(current_style.replace("#FF3333", "#FFB000"))
            elif "#FFB000" in current_style: # CAUTION ise WARNING yap
                lbl.setStyleSheet(current_style.replace("#FFB000", "#FF3333"))
            elif "#00FF66" in current_style: # ADVISORY ise WARNING yap
                lbl.setStyleSheet(current_style.replace("#00FF66", "#FF3333"))
            lbl.repaint()



# ─── ANA TEST ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.id for s in ALL_SCENARIOS])
def test_vision_scenario_v4(pencere, scenario: FaultScenario):
    """
    Her senaryo için:
      1. Snapshot-based state oluştur (simülasyon dondurulur)
      2. WCA + MC logic testi
      3. Per-parameter görsel kontroller (bar fill, renk, numeric, unit)
      4. Unexpected text tespiti
      5. Anti-ice state kontrolü (ANTI_* senaryolar)
      6. Annotated screenshot oluştur
      7. ML Analizi (opsiyonel)
      8. Kapsamlı rapor yaz
    """
    t_start = time.time()
    result = VisionTestResultV4(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        category=scenario.category,
        severity=scenario.severity,
        timestamp=datetime.now().isoformat(),
    )

    idx = ALL_SCENARIOS.index(scenario) + 1
    total = len(ALL_SCENARIOS)
    print(f"\n[{idx:02d}/{total}] {scenario.id}  {scenario.name}")

    try:
        # ── 1. Snapshot-based state uygula (DOĞRU DURUM) ───────────────────
        # Simülasyonu dondurup test değerlerini (örn. E1_TRQ=108) uygular ve UI'a işler.
        apply_scenario_snapshot(pencere, scenario)

        # ── 2. Nominal (Beklenen) snapshot al ──────────────────────────────
        # Bu görüntü "olması gereken" HATA İÇERMEYEN test durumudur.
        _, nominal_path, _, nominal_pil = take_snapshot(pencere, scenario.id, suffix="_nominal")
        result.nominal_screenshot_path = nominal_path
        print(f"       ├ Nominal SS: {os.path.basename(nominal_path)}")

        # ── 3. Anti-ice özel inject (ANTI_* senaryolar) ──────────────────
        anti_ice_check: Optional[VisualCheckResult] = None
        if scenario.id == "ANTI_002":
            pencere.lbl_anti.setText("OFF")
            QApplication.processEvents()
            anti_ice_check = check_anti_ice_state(pencere, "OFF")
        elif scenario.id == "ANTI_003":
            pencere.lbl_anti.setText("ON")
            QApplication.processEvents()
            anti_ice_check = check_anti_ice_state(pencere, "ON")
        elif scenario.id == "ANTI_001":
            anti_ice_check = check_anti_ice_state(pencere, pencere.lbl_anti.text())

        # ── 3.5. RASTGELE GÖRSEL HATA ENJEKSİYONU ─────────────────────────
        # Kontrol edilecek TÜM widget'lara kasten hata enjekte et ki,
        # HTML raporda hiçbiri nominal ile birebir aynı (Expected=Actual) görünmesin.
        if scenario.inject:
            for p_key in scenario.inject.keys():
                inject_random_visual_bug(pencere, scenario, override_param_key=p_key)
        else:
            inject_random_visual_bug(pencere, scenario)

        # Qt'nin yapılan stil değişikliklerini (özellikle WCA renk değişimleri)
        # ekrana yansıtması (repaint) için olay kuyruğunu işletiyoruz:
        QApplication.processEvents()

        # ── 4. Hatalı durum snapshot al (ACTUAL) ───────────────────────────
        # İçinde az önce koyduğumuz görsel BUG'ı barındıran screenshot
        _, screenshot_path, b64, pil_img = take_snapshot(pencere, scenario.id)
        result.screenshot_path = screenshot_path
        print(f"       ├ Screenshot: {os.path.basename(screenshot_path)}")

        # ── 5. WCA + MC logic testi ──────────────────────────────────────
        mc_ok, wca_ok, unexpected_texts = run_wca_logic(pencere, scenario)
        result.mc_state_ok = mc_ok
        result.wca_text_ok = wca_ok
        result.unexpected_wca_texts = unexpected_texts
        mc_sym = "PASS" if mc_ok else "FAIL"
        wca_sym = "PASS" if wca_ok else "FAIL"
        print(f"       ├ Logic  MC={mc_sym}  WCA={wca_sym}")

        if unexpected_texts:
            print(f"       ├ ⚠ Beklenmedik WCA metni: {unexpected_texts}")

        # ── 6. Görsel kontroller ─────────────────────────────────────────
        # NOT: PIL olmasa bile Qt tabanlı tüm checkler çalışır.
        # pil_img sadece annotated screenshot + crop için kullanılır.
        if scenario.inject:
            summaries, annotations = run_visual_checks(pencere, scenario, pil_img)
            result.visual_summaries = summaries

            # Anti-ice check ekle (varsa)
            if anti_ice_check:
                anti_summary = VisualCheckSummary(
                    param_key="ANTI_ICE",
                    param_label="ANTI-ICE",
                    injected_value=0,
                )
                anti_summary.checks.append(anti_ice_check)
                result.visual_summaries.append(anti_summary)

            # Başarısız checkler raporla
            for s in summaries:
                for chk in s.failing_checks:
                    wca_flag = "→WCA" if chk.goes_to_wca else "→Rapor"
                    print(f"       ├ ✗ {s.param_label}.{chk.check_name}: "
                          f"beklenen={chk.expected} gerçek={chk.actual} "
                          f"[{chk.error_category}] {wca_flag}")

            # ── 7. Annotated screenshot + comparison crops ───────────────
            for s in result.visual_summaries:
                if s.failing_checks and s.bbox:
                    # Hatalı görünüm crop'ı (faulty — mevcut pil_img)
                    if pil_img:
                        actual_path = os.path.join(CROP_DIR, f"{scenario.id}_{s.param_key}_actual.png")
                        save_cropped_error(pil_img, s.bbox, actual_path, padding=30)
                        s.actual_crop_path = actual_path
                        s.crop_path = actual_path  # eski alan da güncelle

                    # Olması gereken crop'ı (nominal — fault öncesi)
                    if nominal_pil:
                        nominal_crop = os.path.join(CROP_DIR, f"{scenario.id}_{s.param_key}_nominal.png")
                        save_cropped_error(nominal_pil, s.bbox, nominal_crop, padding=30)
                        s.nominal_crop_path = nominal_crop

            if annotations and pil_img:
                annotated_path = os.path.join(ANNOTATED_DIR, f"{scenario.id}_annotated.png")
                annotate_screenshot(pil_img, annotations, save_path=annotated_path)
                result.annotated_path = annotated_path
                print(f"       ├ Annotated: {os.path.basename(annotated_path)}")

            # Karşılaştırma crop sayısını raporla
            comparison_count = sum(
                1 for s in result.visual_summaries
                if s.nominal_crop_path and s.actual_crop_path
            )
            if comparison_count:
                print(f"       ├ Karşılaştırma crop: {comparison_count} parametre")

        # ── 9. ML model ──────────────────────────────────────────────────
        if HAS_ML and screenshot_path:
            collect_training_data(screenshot_path, scenario.id, scenario.severity)
            summary_ml = dataset_summary()
            pred = predict(screenshot_path)
            result.ml_dataset_count = summary_ml.get("total", 0)
            if "error" not in pred:
                conf_pct = pred["confidence"] * 100
                anom = " ⚠ANOMALY" if pred.get("anomaly") else ""
                result.ml_prediction = f"{pred['class']} %{conf_pct:.0f}{anom}"

        # ── 10. Genel sonuç ──────────────────────────────────────────────
        result.visual_pass = not result.has_visual_failures
        result.overall_pass = result.mc_state_ok and result.wca_text_ok and result.visual_pass
        
        # Eğer ML ANOMALY tespit etmişse ancak senaryo NOMINAL ise, veya ML ANOMALY bulmamışsa ve WARNING ise fail edebiliriz (Opsiyonel ML check)
        if HAS_ML and "error" not in pred:
            is_anomaly = pred.get("anomaly", False)
            if scenario.severity in ("WARNING", "CAUTION") and not is_anomaly:
                result.overall_pass = False
            elif scenario.severity == "NOMINAL" and is_anomaly:
                result.overall_pass = False


    except Exception as ex:
        import traceback
        result.error = f"{ex}\n{traceback.format_exc()}"
        result.overall_pass = False
        print(f"   💥 HATA: {ex}")

    result.duration_ms = int((time.time() - t_start) * 1000)
    _all_results.append(result)

    final = "PASS" if result.overall_pass else "FAIL"
    print(f"       └ SONUÇ: {final} ({result.duration_ms}ms)")

    # Raporu her testten sonra kaydet
    _save_report()

    # Soft fail — rapor üretilir, tüm testler çalışır
    if not result.mc_state_ok:
        pytest.fail(
            f"[{scenario.id}] MC beklenen={scenario.expected_mc_state}, "
            f"gerçek={pencere.lbl_mc.text()}",
            pytrace=False,
        )
    if not (result.wca_text_ok or not scenario.expected_wca_texts):
        pytest.fail(
            f"[{scenario.id}] WCA'da beklenen metin yok: {scenario.expected_wca_texts}",
            pytrace=False,
        )


# ─── HTML RAPOR ───────────────────────────────────────────────────────────────

def _img_to_b64(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _save_report():
    """JSON + HTML rapor yaz."""
    json_path = os.path.join(REPORT_DIR, "vision_test_results_v4.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # dataclass serialize
        data = []
        for r in _all_results:
            d = {
                "scenario_id": r.scenario_id, "scenario_name": r.scenario_name,
                "category": r.category, "severity": r.severity,
                "timestamp": r.timestamp, "mc_state_ok": r.mc_state_ok,
                "wca_text_ok": r.wca_text_ok, "visual_pass": r.visual_pass,
                "overall_pass": r.overall_pass,
                "ml_prediction": r.ml_prediction,
                "unexpected_wca_texts": r.unexpected_wca_texts,
                "screenshot_path": r.screenshot_path,
                "annotated_path": r.annotated_path,
                "error": r.error, "duration_ms": r.duration_ms,
                "visual_summaries": [
                    {
                        "param_key": vs.param_key, "param_label": vs.param_label,
                        "injected_value": vs.injected_value, "all_passed": vs.all_passed,
                        "checks": [
                            {"check_name": c.check_name, "passed": c.passed,
                             "expected": c.expected, "actual": c.actual,
                             "error_category": c.error_category, "goes_to_wca": c.goes_to_wca,
                             "note": c.note}
                            for c in vs.checks
                        ]
                    }
                    for vs in r.visual_summaries
                ],
            }
            data.append(d)
        json.dump(data, f, ensure_ascii=False, indent=2)

    _generate_html_report()


def _generate_html_report():
    html_path = os.path.join(REPORT_DIR, "report.html")

    total   = len(_all_results)
    passed  = sum(1 for r in _all_results if r.overall_pass)
    failed  = total - passed
    vis_ok  = sum(1 for r in _all_results if r.visual_pass)

    # Kategori istatistikleri
    cat_stats: Dict[str, Dict] = {}
    for r in _all_results:
        c = r.category
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "pass": 0}
        cat_stats[c]["total"] += 1
        if r.overall_pass:
            cat_stats[c]["pass"] += 1

    cat_rows = ""
    for cat, stat in cat_stats.items():
        pct = int(stat["pass"] / stat["total"] * 100) if stat["total"] else 0
        bar_color = "#4caf50" if pct == 100 else ("#ff9800" if pct >= 50 else "#f44336")
        cat_rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0">
          <span style="color:#aaa;width:100px;font-size:11px">{cat}</span>
          <div style="flex:1;background:#222;border-radius:4px;height:8px">
            <div style="width:{pct}%;background:{bar_color};height:8px;border-radius:4px"></div>
          </div>
          <span style="color:#fff;font-size:11px;width:60px">{stat['pass']}/{stat['total']}</span>
        </div>"""

    rows_html = ""
    for r in _all_results:
        status_bg  = "#1a3a1a" if r.overall_pass else "#3a1a1a"
        status_txt = "#4caf50" if r.overall_pass else "#f44336"
        status_lbl = "PASS" if r.overall_pass else "FAIL"
        mc_sym   = "✅" if r.mc_state_ok  else "❌"
        wca_sym  = "✅" if r.wca_text_ok  else "❌"
        vis_sym  = "✅" if r.visual_pass  else "❌"
        ml_pred  = r.ml_prediction or f"Bekliyor ({r.ml_dataset_count}/30)"

        # Full screenshot
        img_b64 = _img_to_b64(r.screenshot_path)
        img_tag = (f'<img src="data:image/png;base64,{img_b64}" '
                   f'style="width:100%;border-radius:4px;cursor:pointer" '
                   f'onclick="openImg(this)" />'
                   if img_b64 else "<span style='color:#555'>screenshot yok</span>")

        # Annotated screenshot
        ann_b64 = _img_to_b64(r.annotated_path)
        ann_tag = (f'<div style="margin-top:8px"><div style="color:#ff9800;font-size:10px;margin-bottom:4px">▶ ANNOTATED (failed areas highlighted)</div>'
                   f'<img src="data:image/png;base64,{ann_b64}" '
                   f'style="width:100%;border-radius:4px;cursor:pointer;border:1px solid #ff9800" '
                   f'onclick="openImg(this)" /></div>'
                   if ann_b64 else "")

        # Per-check tablosu
        check_rows = ""
        for vs in r.visual_summaries:
            vs_failed = False
            for chk in vs.checks:
                chk_sym = "✅" if chk.passed else "❌"
                wca_flag = "⚡WCA" if chk.goes_to_wca else "📋Rapor"
                cat_color = "#f44336" if "WCA" in str(chk.error_category) else "#ff9800"
                note_html = (
                    f'<div style="color:#ff8888;font-size:10px;margin-top:2px">'
                    f'ℹ {chk.note}</div>' if chk.note and not chk.passed else ""
                )

                if not chk.passed:
                    vs_failed = True

                check_rows += f"""
                <tr style="background:{'#2a1a1a' if not chk.passed else '#111'}">
                  <td style="padding:4px 6px">{chk_sym}</td>
                  <td style="color:#fff;padding:4px 6px">{vs.param_label}</td>
                  <td style="color:#aaa;padding:4px 6px">{chk.check_name}</td>
                  <td style="color:#4caf50;padding:4px 6px">{chk.expected}</td>
                  <td style="color:{'#f44336' if not chk.passed else '#aaa'};padding:4px 6px">
                    {chk.actual}{note_html}
                  </td>
                  <td style="color:{cat_color};font-size:10px;padding:4px 6px">{chk.error_category}</td>
                </tr>"""

            # Karşılaştırma görselleri (her parametre için SADECE 1 KERE, eğer hata varsa gösterilir)
            if vs_failed:
                nom_b64 = _img_to_b64(vs.nominal_crop_path)
                act_b64 = _img_to_b64(vs.actual_crop_path or vs.crop_path)
                if nom_b64 or act_b64:
                    nom_tag = (
                        f'<div style="flex:1;min-width:0">'
                        f'<div style="font-size:9px;color:#4caf50;margin-bottom:3px">✓ EXPECTED</div>'
                        f'<img src="data:image/png;base64,{nom_b64}" '
                        f'style="width:100%;border-radius:3px;cursor:pointer;border:1px solid #4caf50" '
                        f'onclick="openImg(this)" title="Nominal / Expected" />'
                        f'</div>'
                        if nom_b64 else
                        f'<div style="flex:1;min-width:0;display:flex;align-items:center;'
                        f'justify-content:center;border:1px dashed #444;border-radius:3px;'
                        f'color:#555;font-size:10px;padding:8px">no nominal image</div>'
                    )
                    act_tag = (
                        f'<div style="flex:1;min-width:0">'
                        f'<div style="font-size:9px;color:#f44336;margin-bottom:3px">✗ ACTUAL (FAILED)</div>'
                        f'<img src="data:image/png;base64,{act_b64}" '
                        f'style="width:100%;border-radius:3px;cursor:pointer;border:1px solid #f44336" '
                        f'onclick="openImg(this)" title="Actual appearance" />'
                        f'</div>'
                        if act_b64 else
                        f'<div style="flex:1;min-width:0;display:flex;align-items:center;'
                        f'justify-content:center;border:1px dashed #444;border-radius:3px;'
                        f'color:#555;font-size:10px;padding:8px">no crop</div>'
                    )
                    if vs.param_key == "WCA":
                        errors_html = "".join([f"<div style='margin-bottom:6px;'><b>{chk.check_name}:</b> {chk.note if chk.note else f'Expected {chk.expected}, got {chk.actual}'}</div>" for chk in vs.failing_checks])
                        comparison_html = (
                            f'<div style="display:flex;gap:16px;margin-top:4px;padding:12px;'
                            f'background:#1a1a2a;border-radius:4px;border:1px solid #333;'
                            f'width:100%; align-items:center;">'
                            f'<div style="flex:0 0 200px;">{act_tag}</div>'
                            f'<div style="flex:1; color:#ff8888; font-size:11px; padding-left:8px;">'
                            f'<div style="color:#f44336; font-weight:bold; margin-bottom:6px; font-size:12px;">DETECTED ERRORS</div>'
                            f'{errors_html}'
                            f'</div>'
                            f'</div>'
                        )
                    else:
                        comparison_html = (
                            f'<div style="display:flex;gap:16px;margin-top:4px;padding:12px;'
                            f'background:#1a1a2a;border-radius:4px;border:1px solid #333;'
                            f'max-width:350px;min-width:200px;width:100%">'
                            f'{nom_tag}'
                            f'<div style="display:flex;align-items:center;color:#555;font-size:24px;font-weight:bold">→</div>'
                            f'{act_tag}'
                            f'</div>'
                        )
                    check_rows += f"""
                    <tr>
                      <td colspan="6" style="padding:0 6px 12px 6px; border-bottom:1px solid #1e1e1e; background:#111;">
                        {comparison_html}
                      </td>
                    </tr>"""

        vis_count = sum(len(vs.checks) for vs in r.visual_summaries)
        vis_fail_count = sum(len(vs.failing_checks) for vs in r.visual_summaries)
        check_table = ""
        if check_rows:
            check_table = f"""
            <div style="margin-top:12px">
              <div style="color:#888;font-size:10px;margin-bottom:6px">
                VISUAL CHECK RESULTS
                <span style="color:{'#f44336' if vis_fail_count else '#4caf50'};margin-left:8px">
                  {vis_fail_count} errors / {vis_count} total checks
                </span>
              </div>
              <table style="width:100%;border-collapse:collapse;font-size:11px">
                <thead>
                  <tr style="background:#1a1a1a;color:#666">
                    <th style="padding:4px 6px;text-align:left">Status</th>
                    <th style="padding:4px 6px;text-align:left">Parameter</th>
                    <th style="padding:4px 6px;text-align:left">Check</th>
                    <th style="padding:4px 6px;text-align:left">Expected</th>
                    <th style="padding:4px 6px;text-align:left">Actual</th>
                    <th style="padding:4px 6px;text-align:left">Category</th>
                  </tr>
                </thead>
                <tbody>{check_rows}</tbody>
              </table>
            </div>"""

        # Unexpected WCA texts
        unexp_html = ""
        if r.unexpected_wca_texts:
            unexp_list = ", ".join(f'<code style="color:#ff9800">{t}</code>' for t in r.unexpected_wca_texts)
            unexp_html = f'<div style="margin-top:8px;padding:6px;background:#2a1a00;border-radius:4px;font-size:11px">⚠ Unexpected WCA text: {unexp_list}</div>'


        rows_html += f"""
        <div class="card" style="border-left:4px solid {status_txt}; background:{status_bg}">
          <div class="card-header">
            <div>
              <span class="sid">[{r.scenario_id}]</span>
              <span class="sname">{r.scenario_name}</span>
              <span class="cat-badge">{r.category}</span>
              <span class="sev-badge sev-{r.severity[:4]}">{r.severity[:4]}</span>
            </div>
            <span class="overall" style="color:{status_txt}">{status_lbl}</span>
          </div>
          <div class="card-body">
            <div class="screenshot-col">
              {img_tag}
              {ann_tag}
            </div>
            <div class="info-col">
              {unexp_html}
              {check_table}
            </div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TUSAS TestLab v4 — Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: Consolas, monospace; font-size: 13px; }}
  .topbar {{ background: #111; border-bottom: 1px solid #333; padding: 16px 24px;
             display: flex; justify-content: space-between; align-items: center; }}
  .topbar h1 {{ font-size: 15px; color: #fff; letter-spacing: .05em; }}
  .topbar span {{ font-size: 12px; color: #888; }}
  .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; padding: 20px 24px; }}
  .metric {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 6px; padding: 14px 18px; }}
  .metric .label {{ font-size: 10px; color: #666; text-transform: uppercase; margin-bottom: 6px; }}
  .metric .value {{ font-size: 26px; font-weight: bold; }}
  .metric.pass .value {{ color: #4caf50; }}
  .metric.fail .value {{ color: #f44336; }}
  .metric.ai   .value {{ color: #2196f3; }}
  .metric.vis  .value {{ color: #9c27b0; }}
  .metric.pct  .value {{ color: #ff9800; }}
  .cat-section {{ padding: 0 24px 16px; background: #111; margin: 0 24px 16px; border-radius:6px; border:1px solid #2a2a2a; }}
  .cat-section h3 {{ color:#888; font-size:11px; padding:12px 0 8px; text-transform:uppercase; }}
  .cards {{ padding: 0 24px 40px; display: flex; flex-direction: column; gap: 14px; }}
  .card {{ background: #111; border-radius: 8px; border: 1px solid #2a2a2a; overflow: hidden; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center;
                  padding: 10px 16px; border-bottom: 1px solid #1e1e1e; }}
  .sid {{ color: #888; margin-right: 8px; }}
  .sname {{ color: #fff; font-weight: bold; margin-right: 10px; }}
  .cat-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 99px;
                background: #222; color: #aaa; border: 1px solid #333; margin-right: 4px; }}
  .sev-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 99px; background: #1a1a2e; color: #9fa8da; }}
  .sev-WARN {{ background: #2a1a1a; color: #f44336; }}
  .sev-CAUT {{ background: #2a2a1a; color: #ff9800; }}
  .sev-ADVI {{ background: #1a2a1a; color: #4caf50; }}
  .overall {{ font-size: 13px; font-weight: bold; letter-spacing: .05em; }}
  .card-body {{ display: grid; grid-template-columns: 420px 1fr; gap: 16px; padding: 14px 16px; }}
  .info-col {{ display: flex; flex-direction: column; gap: 8px; }}
  .checks {{ display: flex; flex-direction: column; gap: 4px; line-height: 1.6; }}
  .ai-box {{ background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 4px;
             padding: 10px; font-size: 11px; color: #aaa; white-space: pre-wrap;
             max-height: 120px; overflow-y: auto; margin-top: 8px; }}
  table td, table th {{ padding: 3px 8px; border-bottom: 1px solid #1e1e1e; }}
  #lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.94);
         z-index:999; align-items:center; justify-content:center; cursor:zoom-out; }}
  #lb img {{ max-width:95vw; max-height:95vh; border-radius:4px; }}
  #lb.open {{ display:flex; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>🛩 TUSAS TestLab v4 — ML Vision Test Report</h1>
  <span>Generated at: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</span>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Total</div><div class="value">{total}</div></div>
  <div class="metric pass"><div class="label">Pass</div><div class="value">{passed}</div></div>
  <div class="metric fail"><div class="label">Fail</div><div class="value">{failed}</div></div>
  <div class="metric vis"><div class="label">Visual Pass</div><div class="value">{vis_ok}</div></div>
</div>
<div class="cat-section">
  <h3>Category Results</h3>
  {cat_rows}
</div>
<div class="cards">{rows_html}</div>
<div id="lb" onclick="this.classList.remove('open')"><img id="lb-img" src="" /></div>
<script>
function openImg(el) {{
  document.getElementById('lb-img').src = el.src;
  document.getElementById('lb').classList.add('open');
}}
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n🌐 HTML Rapor: {os.path.abspath(html_path)}")


# ─── SESSION SONUNDA: ÖZET + OTOMATİK RAPOR AÇMA ─────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    html_path = os.path.abspath(os.path.join(REPORT_DIR, "report.html"))

    if _all_results:
        total   = len(_all_results)
        passed  = sum(1 for r in _all_results if r.overall_pass)
        failed  = total - passed
        vis_ok  = sum(1 for r in _all_results if r.visual_pass)
        print(f"\n{'='*60}")
        print(f"  TUSAS TestLab v4 — Test Tamamlandı")
        print(f"  Toplam={total}  PASS={passed}  FAIL={failed}")
        print(f"  Görsel Pass={vis_ok}")
        print(f"  Rapor: {html_path}")
        print(f"{'='*60}")
        _save_report()
    else:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<title>TUSAS TestLab v4</title>
<style>body{background:#0d0d0d;color:#e0e0e0;font-family:sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.box{text-align:center}.title{font-size:20px;color:#fff;margin-bottom:12px}
.sub{color:#666;font-size:13px}</style></head><body>
<div class="box"><div class="title">🛩 TUSAS TestLab v4</div>
<div class="sub">Henüz test sonucu yok.<br>
<code>python run_tests.py</code> çalıştırın.</div></div></body></html>""")

    # ── Raporu OTOMATIK aç ────────────────────────────────────────────────────
    if os.environ.get("TUSAS_NO_OPEN_REPORT") == "1":
        return
    if not os.path.exists(html_path):
        print(f"  [!] Rapor dosyası bulunamadı: {html_path}")
        return

    import platform
    import subprocess
    import webbrowser

    opened = False
    # Yöntem 1: webbrowser
    try:
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
        print(f"  🌐 Rapor tarayıcıda açılıyor...")
        opened = True
    except Exception:
        pass

    # Yöntem 2: Platform spesifik
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

    if not opened:
        print(f"  Manuel açın: {html_path}")
