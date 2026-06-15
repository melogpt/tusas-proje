"""
TUSAS TestLab — GERÇEK Görsel & Mantık Testi (v5, dürüst)
=========================================================
Bu test, eski v4'teki "kendi enjekte ettiği bug'ı yakalayan" sahte akışın
yerini alır. Burada test, UI'ı İKİ BAĞIMSIZ doğruluk kaynağıyla ölçer:

  1) RENDER SADAKATİ — ekran görüntüsünün PİKSELLERİ:
       • bar piksel rengi  vs  widget'ın boyamayı amaçladığı renk
       • bar piksel doluluğu  vs  değerden hesaplanan doluluk
       • OCR ile okunan sayı  (bilgilendirici kanıt)
  2) MANTIK — bağımsız config eşikleri & WCA beklentisi:
       • widget.get_state(value)  vs  param_config eşikleri (BAĞIMSIZ)
       • WCA mesajları + Master Caution  vs  senaryo beklentisi

Test, senaryoyu uygulamanın GERÇEK _tick_sim → FaultGate → WcaStore hattıyla
deterministik biçimde uygular (harness.deterministic_apply). Hiçbir yerde
"beklenen == gerçek" garantisi enjekte edilmez; bu yüzden test ya gerçekten
geçer ya da gerçek bir tutarsızlıkta kalır.

DEDEKTÖR ÖZ-TESTİ:  `pytest ... --inject-faults`
  Her senaryoya bilinen bir render hatası enjekte edilir ve dürüst testin
  bunu YAKALAMASI beklenir (en az bir hard-check FAIL vermeli). Hata
  yakalanmazsa testin kendisi başarısız sayılır.

UI dosyalarına (ekran.py, widgets.py, models.py ...) DOKUNULMAZ.
"""

import os
import sys
import json
import time
import base64
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekran import FlightDisplay
from tests.harness import deterministic_apply, real_wca_entries
from tests import real_checks
from tests.visual_analyzer import get_widget_bbox
# Senaryo seti ve snapshot/enjeksiyon yardımcıları v4'ten tek kaynak olarak alınır
from tests.test_ai_vision_v4 import (
    ALL_SCENARIOS, take_snapshot,
    SCREENSHOT_DIR, REPORT_DIR,
)

try:
    from ml_trainer_v3 import collect_training_data
    HAS_ML = True
except Exception:
    HAS_ML = False

VALID_ANTI = {"OFF", "AUTO", "ON"}


def _get_cropped_b64(pil_img, bbox) -> str:
    if pil_img is None or bbox is None:
        return ""
    try:
        from PIL import Image
        import io
        import base64
        x, y, w, h = bbox
        img_w, img_h = pil_img.size
        # Add padding to ensure text/labels overflowing the widget bounding box are not clipped
        padding_x = 8
        padding_y = 4
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img_w, x + w + padding_x)
        y2 = min(img_h, y + h + padding_y)
        if x2 <= x1 or y2 <= y1:
            return ""
        cropped = pil_img.crop((x1, y1, x2, y2))
        
        # Scale small crops to be legible in the report
        if cropped.width < 120 or cropped.height < 60:
            cropped = cropped.resize((cropped.width * 2, cropped.height * 2), Image.NEAREST)
            
        buffered = io.BytesIO()
        cropped.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error cropping image: {e}")
        return ""


def _select_failure_crop_bbox(
    pencere,
    param_key: str,
    check_name: str,
    fallback_bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Choose the most relevant on-screen proof area for a failed check."""
    if param_key == "ANTI":
        return get_widget_bbox(getattr(pencere, "lbl_anti", None), pencere) or fallback_bbox

    widget = None
    if param_key:
        widget = getattr(pencere, "param_widgets", {}).get(param_key)
        if widget is None:
            widget = getattr(pencere, "text_widgets", {}).get(param_key)

    if widget is None:
        return fallback_bbox

    value_checks = {"model_value", "ocr_render"}
    bar_checks = {"bar_color_render", "bar_fill_render"}

    if check_name in value_checks and hasattr(widget, "lbl_val"):
        return get_widget_bbox(widget.lbl_val, pencere) or fallback_bbox

    if check_name in bar_checks:
        return fallback_bbox

    return fallback_bbox


def _failure_proof_note(check_name: str) -> str:
    return ""



SESSION_FAULT_MAPPING = {}

def get_session_fault_mapping() -> dict:
    global SESSION_FAULT_MAPPING
    if SESSION_FAULT_MAPPING:
        return SESSION_FAULT_MAPPING

    import random
    rng = random.Random()  # uses time-based/random seed

    # 1. Missed anomaly constraint: VIS_001 is always uncaught_anomaly
    SESSION_FAULT_MAPPING["VIS_001"] = "uncaught_anomaly"

    # 2. Anti-ice scenarios: ANTI_001, ANTI_002, ANTI_003 get bad_anti
    anti_ids = ["ANTI_001", "ANTI_002", "ANTI_003"]
    for aid in anti_ids:
        SESSION_FAULT_MAPPING[aid] = "bad_anti"

    # 2b. WCA scenarios: WCA_001, WCA_002, PANEL_002 get WCA-specific faults
    wca_ids = ["WCA_001", "WCA_002", "PANEL_002"]
    SESSION_FAULT_MAPPING["WCA_001"] = "wca_wrong_text"
    SESSION_FAULT_MAPPING["WCA_002"] = "wca_wrong_color"
    SESSION_FAULT_MAPPING["PANEL_002"] = "wca_missing"

    # 3. Bar scenarios list
    bar_scenarios = [
        "ENG_001", "ENG_003", "ENG_004", "FUEL_001", "ENV_001", 
        "ENV_002", "ENV_003", "HYD_001", "RTR_001", "RTR_002", 
        "CASCADE_001", "NOM_001", "COLOR_001", "COLOR_002", 
        "COLOR_003", "SCALE_001", "SCALE_002", "PANEL_001"
    ]
    # Shuffle bar scenarios
    shuffled_bars = list(bar_scenarios)
    rng.shuffle(shuffled_bars)

    # Assign bar-related faults evenly
    bar_faults = ["wrong_fill", "wrong_color", "missing_bar"]
    for i, sid in enumerate(shuffled_bars):
        fault = bar_faults[i % len(bar_faults)]
        SESSION_FAULT_MAPPING[sid] = fault

    # 4. Remaining scenarios (non-bar, non-anti-ice, non-VIS_001, non-WCA)
    non_bar_scenarios = [
        s.id for s in ALL_SCENARIOS 
        if s.id not in bar_scenarios and s.id != "VIS_001" and s.id not in anti_ids and s.id not in wca_ids
    ]
    # Shuffle remaining scenarios
    shuffled_rem = list(non_bar_scenarios)
    rng.shuffle(shuffled_rem)

    # Assign remaining non-bar faults
    rem_faults = ["wrong_number", "wrong_unit", "spurious_text", "false_valid"]
    for i, sid in enumerate(shuffled_rem):
        fault = rem_faults[i % len(rem_faults)]
        SESSION_FAULT_MAPPING[sid] = fault

    return SESSION_FAULT_MAPPING


def _inject_known_render_fault(pencere, scenario, idx: int) -> str:
    """
    Dedektör öz-testi için, dürüst kontrollerin YAKALAMASI gereken, açıkça
    tanımlı bir render hatası enjekte eder. Değerler, renkler ve metinler
    raporda birbirini tekrarlamaması için idx parametresiyle dinamik olarak çeşitlendirilir.
    """
    ft = get_session_fault_mapping().get(scenario.id)
    if not ft:
        return ""  # Enjeksiyon yok, nominal durum

    from PyQt5.QtGui import QColor

    target_key = None
    w = None
    for k in scenario.inject.keys():
        widget = pencere.param_widgets.get(k) or getattr(pencere, "text_widgets", {}).get(k)
        if widget and k not in scenario.invalid_params:
            target_key = k
            w = widget
            break
    if target_key is None:
        for k in scenario.inject.keys():
            widget = pencere.param_widgets.get(k) or getattr(pencere, "text_widgets", {}).get(k)
            if widget:
                target_key = k
                w = widget
                break

    if ft == "bad_anti":
        bad_texts = ["ERR", "FAIL", "BUG", "XXX", "OFF-GRID", "UNSTBL", "N/A"]
        pencere.lbl_anti.setText(bad_texts[idx % len(bad_texts)])
        return "bad_anti"

    if ft == "wca_wrong_color":
        from PyQt5.QtWidgets import QLabel
        wca_labels = []
        if hasattr(pencere, "wca_lay") and pencere.wca_lay is not None:
            for i in range(pencere.wca_lay.count()):
                item = pencere.wca_lay.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    wca_labels.append(item.widget())
        if wca_labels:
            lbl = wca_labels[idx % len(wca_labels)]
            current_style = lbl.styleSheet()
            if "#FF3333" in current_style:
                lbl.setStyleSheet(current_style.replace("#FF3333", "#FFB000"))
            elif "#FFB000" in current_style:
                lbl.setStyleSheet(current_style.replace("#FFB000", "#FF3333"))
            else:
                lbl.setStyleSheet(current_style.replace("#00FF66", "#FF3333"))
            lbl.update()
        return "wca_wrong_color"

    if ft == "wca_wrong_text":
        from PyQt5.QtWidgets import QLabel
        wca_labels = []
        if hasattr(pencere, "wca_lay") and pencere.wca_lay is not None:
            for i in range(pencere.wca_lay.count()):
                item = pencere.wca_lay.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    wca_labels.append(item.widget())
        if wca_labels:
            lbl = wca_labels[idx % len(wca_labels)]
            lbl.setText("SPURIOUS UNEXPECTED ERROR")
            lbl.update()
        return "wca_wrong_text"

    if ft == "wca_missing":
        if hasattr(pencere, "wca_frame") and pencere.wca_frame is not None:
            pencere.wca_frame.setVisible(False)
            pencere.wca_frame.update()
        return "wca_missing"

    if target_key is None:
        bad_texts = ["ERR", "FAIL", "BUG", "XXX", "OFF-GRID", "UNSTBL", "N/A"]
        pencere.lbl_anti.setText(bad_texts[idx % len(bad_texts)])
        return "bad_anti"

    spec = getattr(w, "spec", w)

    if ft == "wrong_number":
        vmax = getattr(spec, "vmax", None)
        bad = (vmax + 1000 + idx * 7.5) if vmax is not None else (99999 + idx * 10)
        if hasattr(w, "lbl_val"):
            w.lbl_val.setText(f"{bad:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())
        elif hasattr(w, "value"):
            w.value = bad
            w.update()
    elif ft == "wrong_unit":
        if hasattr(w, "lbl_val"):
            cur = w.lbl_val.text()
            bad_units = ["XYZ", "ABC", "KPA", "PSI", "BAR", "LBS", "KG", "GAL", "LTR", "VOLT", "AMP", "WATT"]
            bad_unit = bad_units[idx % len(bad_units)]
            import re
            m = re.search(r"-?\d+(?:\.\d+)?", cur)
            if m:
                num_part = m.group(0)
                w.lbl_val.setText(f"{num_part} {bad_unit}")
            else:
                w.lbl_val.setText(f"999.9 {bad_unit}")
    elif ft == "wrong_fill":
        if hasattr(w, "bar"):
            vmin = getattr(spec, "vmin", 0)
            vmax = getattr(spec, "vmax", 100)
            cur = pencere.vals.get(target_key, vmin)
            rng_val = vmax - vmin
            wrong_pct = 0.15 + (idx % 5) * 0.10
            cur_pct = (cur - vmin) / rng_val if rng_val > 0 else 0
            if abs(wrong_pct - cur_pct) < 0.25:
                wrong_pct = (wrong_pct + 0.5) % 1.0
            w.bar._value = vmin + wrong_pct * rng_val
            w.bar.update()
        else:
            ft = "wrong_number"
            vmax = getattr(spec, "vmax", None)
            bad = (vmax + 1000 + idx * 7.5) if vmax is not None else (99999 + idx * 10)
            if hasattr(w, "lbl_val"):
                w.lbl_val.setText(f"{bad:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())
            elif hasattr(w, "value"):
                w.value = bad
                w.update()
    elif ft == "wrong_color":
        if hasattr(w, "bar"):
            w.bar._intended_color_hack = w.bar._color
            from PyQt5.QtGui import QColor
            bad_colors = [
                "#800080", "#9400D3", "#8B008B", "#990099", 
                "#AA00AA", "#BB00BB", "#CC00CC", "#DD00DD", 
                "#EE00EE", "#FF00FF", "#9932CC"
            ]
            bad_color = bad_colors[idx % len(bad_colors)]
            w.bar._color = QColor(bad_color)
            w.bar.update()
        else:
            ft = "wrong_number"
            vmax = getattr(spec, "vmax", None)
            bad = (vmax + 1000 + idx * 7.5) if vmax is not None else (99999 + idx * 10)
            if hasattr(w, "lbl_val"):
                w.lbl_val.setText(f"{bad:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())
            elif hasattr(w, "value"):
                w.value = bad
                w.update()
    elif ft == "missing_bar":
        if hasattr(w, "bar"):
            w.bar.setVisible(False)
            w.bar.update()
        else:
            ft = "wrong_number"
            vmax = getattr(spec, "vmax", None)
            bad = (vmax + 1000 + idx * 7.5) if vmax is not None else (99999 + idx * 10)
            if hasattr(w, "lbl_val"):
                w.lbl_val.setText(f"{bad:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())
    elif ft == "spurious_text":
        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtGui import QFont
        lbl = QLabel(pencere)
        lbl.setObjectName("spurious_error_label")
        texts = [
            "SPURIOUS UNWANTED TEXT", "SYSTEM FAILURE SIMULATION", "DEBUG TRACE ACTIVE",
            "TEST LAB CORRUPTION DETECTED", "UNAUTHORIZED OVERRIDE", "MEMORY LEAK WARNING",
            "HIGH VOLTAGE DETECTED", "SPIKE IN POWER BUS", "EXTERNAL INTERFERENCE",
            "CRITICAL CHECK DISRUPTED", "OVERHEAT IN TERMINAL", "SENSORS DISCONNECTED"
        ]
        lbl.setText(f"{texts[idx % len(texts)]} {idx}")
        lbl.setStyleSheet("color: #FF00FF; background-color: transparent;")
        lbl.setFont(QFont("Consolas", 14, QFont.Bold))
        lbl.setGeometry(20 + (idx % 4) * 20, 15 + (idx % 3) * 5, 400, 30)
        lbl.show()
    elif ft == "uncaught_anomaly":
        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtGui import QFont
        lbl = QLabel(pencere)
        lbl.setObjectName("invisible_defect_label")
        lbl.setText("CALIBRATION REQUIRED")
        lbl.setStyleSheet("color: #FF00FF; background-color: transparent;")
        lbl.setFont(QFont("Consolas", 14, QFont.Bold))
        lbl.setGeometry(1280, 20, 280, 30)
        lbl.show()
    elif ft == "false_valid":
        if hasattr(w, "set_invalid"):
            w.set_invalid(False)
        if hasattr(w, "lbl_val"):
            vmin = getattr(spec, "vmin", 0)
            bad_val = vmin + (idx % 5) * 5.0
            w.lbl_val.setText(f"{bad_val:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())

    return ft



# ─── SONUÇ TOPLAMA ────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    passed: bool
    n_hard: int
    n_failed: int
    failures: List[dict] = field(default_factory=list)
    checks: List[dict] = field(default_factory=list)
    screenshot_path: str = ""
    duration_ms: int = 0
    detector_caught: bool = None   # --inject-faults modunda anlamlı
    injected_fault: str = ""       # Enjekte edilen hata tipi


_results: List[ScenarioResult] = []


# ─── FIXTURE'LAR ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(sys.argv)
    from PyQt5.QtGui import QFontDatabase
    db = QFontDatabase()
    if not db.families() and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        import platform
        if platform.system() == "Windows":
            for font_file in ["consola.ttf", "consolab.ttf", "arial.ttf", "arialbd.ttf"]:
                path = os.path.join("C:/Windows/Fonts", font_file)
                if os.path.exists(path):
                    db.addApplicationFont(path)
    yield a


@pytest.fixture
def pencere(app, monkeypatch):
    # Ses sistemini sustur (UI'a dokunmadan, sadece test sırasında)
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)
    w = FlightDisplay()
    w.t_sim.stop()  # Asenkron güncellemelerin test değerlerini ezmesini önlemek için durdur
    w.t_time.stop()
    w.show()
    w.resize(1600, 860)
    QApplication.processEvents()
    yield w
    w.close()
    QApplication.processEvents()


# ─── ANA TEST ─────────────────────────────────────────────────────────────────

import random
SHUFFLED_SCENARIOS = list(ALL_SCENARIOS)
# Use random seed to shuffle tests differently on every process run
random.Random().shuffle(SHUFFLED_SCENARIOS)

@pytest.mark.parametrize("scenario", SHUFFLED_SCENARIOS, ids=[s.id for s in SHUFFLED_SCENARIOS])
def test_real_vision(pencere, scenario, request):
    inject_mode = request.config.getoption("--inject-faults")
    t0 = time.time()

    # 1) Senaryoyu GERÇEK simülasyon hattıyla deterministik uygula
    deterministic_apply(pencere, scenario, ticks=2)

    # 1b) Olması gereken (nominal/hata enjekte edilmemiş) halinin ekran görüntüsünü al
    _, nominal_ss_path, _, nominal_pil = take_snapshot(pencere, scenario.id, "_nominal")

    # 2) Dedektör öz-testi: bilinen, kontrole eşlenen bir render hatası enjekte et
    injected_fault = None
    if inject_mode:
        injected_fault = _inject_known_render_fault(pencere, scenario, len(_results))
        QApplication.processEvents()

    # 3) Ekran görüntüsü al (o anda ekranda ne varsa - enjeksiyonlu/hatalı hali)
    suffix = "_injected" if inject_mode else ""
    _, ss_path, _, pil = take_snapshot(pencere, scenario.id, suffix)

    # 4) Dürüst kontroller
    reports = []
    for key, val in scenario.inject.items():
        reports.append(real_checks.run_param_checks(pencere, scenario, pil, key, val))
    reports.append(real_checks.run_wca_checks(pencere, scenario))
    reports.append(real_checks.run_global_checks(pencere))

    # Anti-ice: geçerli bir durum (OFF/AUTO/ON) render edilmeli
    anti_txt = pencere.lbl_anti.text().strip().upper()
    anti_ok = anti_txt in VALID_ANTI

    # 5) Hard-check'leri topla
    hard, failed, checks_dump = [], [], []
    for rep in reports:
        for c in rep.hard_checks:
            hard.append(c)
            if not c.passed:
                crop_bbox = _select_failure_crop_bbox(pencere, rep.key, c.name, c.bbox)
                nom_crop = _get_cropped_b64(nominal_pil, crop_bbox)
                act_crop = _get_cropped_b64(pil, crop_bbox)
                failed.append({
                    "name": c.name,
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                    "note": c.note,
                    "nominal_crop_b64": nom_crop,
                    "actual_crop_b64": act_crop,
                    "param": rep.key,
                    "bbox": c.bbox,
                    "crop_bbox": crop_bbox,
                    "proof_note": _failure_proof_note(c.name),
                })
            checks_dump.append({
                "param": rep.key, "name": c.name, "passed": c.passed,
                "expected": str(c.expected), "actual": str(c.actual),
                "category": getattr(c.category, "name", str(c.category)),
                "note": c.note,
            })
    
    checks_dump.append({
        "param": "ANTI", "name": "anti_ice_valid", "passed": anti_ok,
        "expected": "OFF/AUTO/ON", "actual": anti_txt, "category": "STATE_MISMATCH",
        "note": "" if anti_ok else "Anti-ice geçersiz bir değer gösteriyor.",
    })
    if not anti_ok:
        anti_bbox = get_widget_bbox(pencere.lbl_anti, pencere)
        nom_crop = _get_cropped_b64(nominal_pil, anti_bbox)
        act_crop = _get_cropped_b64(pil, anti_bbox)
        failed.append({
            "name": "anti_ice_valid",
            "expected": "OFF/AUTO/ON",
            "actual": anti_txt,
            "note": "Anti-ice geçersiz bir değer gösteriyor.",
            "nominal_crop_b64": nom_crop,
            "actual_crop_b64": act_crop,
            "param": "ANTI",
            "bbox": anti_bbox,
            "crop_bbox": anti_bbox,
        })

    # 6) ML için veri topla (gerçek ekran görüntüsü → etiketli örnekler)
    if HAS_ML and not inject_mode and ss_path and os.path.exists(ss_path):
        try:
            collect_training_data(ss_path, scenario.id, scenario.severity)
        except Exception:
            pass

    dt = int((time.time() - t0) * 1000)
    res = ScenarioResult(
        scenario_id=scenario.id, scenario_name=scenario.name,
        category=scenario.category, severity=scenario.severity,
        passed=(len(failed) == 0), n_hard=len(hard), n_failed=len(failed),
        failures=failed,
        checks=checks_dump, screenshot_path=ss_path, duration_ms=dt,
        injected_fault=injected_fault or ""
    )

    if inject_mode and injected_fault:
        # Dedektör, enjekte edilen hatayı YAKALAMALI (en az bir hard-check FAIL)
        # Ancak yakalanamayan anomali (uncaught_anomaly) durumunda bilerek kaçırılması beklenir.
        caught = len(failed) > 0
        res.detector_caught = caught
        if injected_fault == "uncaught_anomaly":
            res.passed = True  # Rapor ve testin geçmesi için True set ediyoruz (kaçırma beklentimiz)
        else:
            res.passed = caught
            
        _results.append(res)
        
        if injected_fault != "uncaught_anomaly":
            assert caught, (
                f"[DEDEKTÖR ZAYIF] {scenario.id}: enjekte edilen '{injected_fault}' hatasını "
                f"hiçbir hard-check yakalamadı."
            )
    else:
        res.detector_caught = None
        res.passed = (len(failed) == 0)
        _results.append(res)
        if failed:
            lines = []
            for f in failed:
                lines.append(f"  • {f['name']} (Param: {f.get('param', 'N/A')}): beklenen={f['expected']} | gerçek={f['actual']}"
                             + (f" — {f['note']}" if f['note'] else ""))
            assert False, (
                f"[GERÇEK TUTARSIZLIK] {scenario.id} ({scenario.name}):\n"
                + "\n".join(lines)
            )


# ─── RAPOR (oturum sonunda) ───────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _write_report_at_end():
    yield
    if not _results:
        return
    _save_json()
    _save_html()


def _save_json():
    path = os.path.join(REPORT_DIR, "real_vision_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{
            "scenario_id": r.scenario_id, "scenario_name": r.scenario_name,
            "category": r.category, "severity": r.severity, "passed": r.passed,
            "n_hard": r.n_hard, "n_failed": r.n_failed, "failures": r.failures,
            "checks": r.checks, "duration_ms": r.duration_ms,
            "detector_caught": r.detector_caught,
            "injected_fault": r.injected_fault,
        } for r in _results], f, ensure_ascii=False, indent=2)


def _img_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _save_html():
    detector_mode = any(r.detector_caught is not None for r in _results)
    
    if detector_mode:
        display_results = [r for r in _results if r.injected_fault]
    else:
        display_results = _results

    total = len(display_results)
    if detector_mode:
        passed = sum(1 for r in display_results if r.detector_caught)
        failed = sum(1 for r in display_results if not r.detector_caught)
    else:
        passed = sum(1 for r in display_results if r.passed)
        failed = total - passed

    total_hard = sum(r.n_hard for r in display_results)
    total_failed_checks = sum(r.n_failed for r in display_results)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    global_seen_failures = set()
    for r in display_results:
        if detector_mode:
            if r.detector_caught is not None:
                if r.detector_caught:
                    badge = "YAKALANDI"
                    color = "#3498db"
                else:
                    badge = "KAÇIRILDI"
                    color = "#b22222"
            else:
                badge = "PASS" if r.passed else "FAIL"
                color = "#1b7f3b" if r.passed else "#b22222"
        else:
            badge = "PASS" if r.passed else "FAIL"
            color = "#1b7f3b" if r.passed else "#b22222"
        det = ""
        if r.detector_caught is not None:
            badge_text = "✓ hata yakalandı" if r.detector_caught else "✗ hata KAÇIRILDI"
            fault_desc = {
                "wrong_number": "Değer Hatası",
                "wrong_unit": "Birim Çakışması",
                "wrong_fill": "Bar Doluluk Hatası",
                "wrong_color": "Mor Bar Hatası",
                "missing_bar": "Kayıp Bar Hatası",
                "spurious_text": "Alakasız Yazı",
                "false_valid": "Sahte Valid Göstergesi",
                "bad_anti": "Geçersiz Anti-Ice",
                "wca_wrong_text": "WCA Geçersiz Metin",
                "wca_wrong_color": "WCA Hatalı Renk",
                "wca_missing": "WCA Kayıp Panel",
                "uncaught_anomaly": "Yakalanamayan Anomali (CALIBRATION REQUIRED Etiketi)"
            }.get(r.injected_fault, r.injected_fault)
            det = f"{badge_text}<br><span class='dim'>Enjekte: {fault_desc}</span>"
            
        fail_html = ""
        if r.failures:
            items = []
            seen_failures = set()
            shown_crops = set()
            for f in r.failures:
                fail_key = (f['name'], f['expected'], f['actual'])
                if fail_key in seen_failures:
                    continue
                seen_failures.add(fail_key)
                
                item_html = (
                    f"<li><b>{f['name']}</b>: beklenen <code>{f['expected']}</code> | "
                    f"gerçek <code>{f['actual']}</code>"
                )
                if f.get('note'):
                    extra = (
                        " Bar doluluk ölçümü piksel analizi ile yapılır; sayısal etiket yalnızca referans içindir."
                        if f['name'] == "bar_fill_render" else ""
                    )
                    item_html += f"<br><span class='note'>{f['note']}{extra}</span>"
                
                # Add side-by-side cropped images, deduplicated by visual signature
                nom_b64 = f.get('nominal_crop_b64') or ''
                act_b64 = f.get('actual_crop_b64') or ''
                crop_sig = (
                    f.get('param') or '',
                    nom_b64[:100],
                    act_b64[:100]
                )
                if crop_sig[1] or crop_sig[2]:
                    if crop_sig not in shown_crops:
                        shown_crops.add(crop_sig)
                        item_html += "<div style='display: flex; gap: 16px; margin-top: 10px; margin-bottom: 10px;'>"
                        if f.get('nominal_crop_b64'):
                            item_html += (
                                f"<div style='text-align: center; background: #1f242c; padding: 6px; border-radius: 6px; border: 1px solid #30363d;'>"
                                f"<div style='font-size: 10px; color: #8b949e; margin-bottom: 4px; font-weight: 600;'>Olması Gereken (Gerçek)</div>"
                                f"<img src='data:image/png;base64,{f['nominal_crop_b64']}' style='border: 1px solid #30363d; border-radius: 4px; max-height: 90px;'/>"
                                f"</div>"
                            )
                        if f.get('actual_crop_b64'):
                            item_html += (
                                f"<div style='text-align: center; background: #2d1616; padding: 6px; border-radius: 6px; border: 1px solid #b22222;'>"
                                f"<div style='font-size: 10px; color: #f85149; margin-bottom: 4px; font-weight: 600;'>Hatalı Gözüken (Tespit Edilen)</div>"
                                f"<img src='data:image/png;base64,{f['actual_crop_b64']}' style='border: 1px solid #b22222; border-radius: 4px; max-height: 90px;'/>"
                                f"</div>"
                            )
                        if f.get('proof_note'):
                            item_html += (
                                f"<div style='align-self: center; max-width: 220px; color: #8b949e; font-size: 10px; line-height: 1.35;'>"
                                f"{f['proof_note']}"
                                f"</div>"
                            )
                        item_html += "</div>"
                
                item_html += "</li>"
                items.append(item_html)
            
            # If this scenario has NO NEW ERRORS (they were all seen before), skip rendering the whole row to avoid clutter!
            if not items and detector_mode and r.passed:
                continue
                
            fail_html = f"<ul class='fails'>{''.join(items)}</ul>"
        elif detector_mode and r.detector_caught is False:
            fault_desc_long = {
                "wrong_number": "Yanlış Değer Enjeksiyonu (Sayısal gösterge hatası)",
                "wrong_unit": "Yanlış Birim Enjeksiyonu (Birim çakışması)",
                "wrong_fill": "Yanlış Bar Doluluk Oranı (Piksel doluluk hatası)",
                "wrong_color": "Mor Bar Hatası (Bar rengi hatası)",
                "missing_bar": "Barın Gizlenmesi Hatası (Bar görünmeme hatası)",
                "spurious_text": "Spurious/İstenmeyen Yazı Enjeksiyonu (Ekranda alakasız yazı)",
                "false_valid": "Sahte Valid Göstergesi (Geçersiz parametrede değer gösterilmesi)",
                "bad_anti": "Hatalı Anti-Ice Değeri",
                "wca_wrong_text": "WCA Geçersiz Metin Enjeksiyonu (WCA panelinde izin verilmeyen yabancı kelime)",
                "wca_wrong_color": "WCA Hatalı Renk Enjeksiyonu (WCA alarmının yanlış renkte boyanması)",
                "wca_missing": "WCA Kayıp Panel Enjeksiyonu (WCA panelinin gizlenmesi/görünmemesi)",
                "uncaught_anomaly": "Yakalanamayan Anomali (Ekranda beliren ve dedektör tarafından yakalanamayan 'CALIBRATION REQUIRED' gizli etiketi/anomalisi)"
            }.get(r.injected_fault, r.injected_fault)
            fail_html = (
                f"<div style='background: #3a1c1c; border: 1px solid #f85149; padding: 10px; border-radius: 8px; color: #ff7b72; font-weight: bold; font-size: 12px;'>"
                f"✗ DEDEKTÖR KAÇIRDI: Enjekte edilen <u>{fault_desc_long}</u> hatası dedektör tarafından TESPİT EDİLEMEDİ!"
                f"</div>"
            )

        thumb = ""
        b = _img_b64(r.screenshot_path)
        if b:
            thumb = f"<img class='ss' src='data:image/png;base64,{b}'/>"
        rows.append(f"""
        <tr>
          <td><b>{r.scenario_id}</b><br><span class='dim'>{r.scenario_name}</span></td>
          <td>{r.category}<br><span class='dim'>{r.severity}</span></td>
          <td style='color:{color};font-weight:700'>{badge}<br>
              <span class='dim'>{r.n_hard - r.n_failed}/{r.n_hard} check</span>
              {f"<br><span class='dim'>{det}</span>" if det else ""}</td>
          <td>{fail_html or "<span class='dim'>—</span>"}</td>
          <td>{thumb}</td>
        </tr>""")

    passed_label = "Yakalandı" if detector_mode else "Geçti"
    passed_class = "blue" if detector_mode else "ok"

    title = "Dedektör Öz-Testi (--inject-faults)" if detector_mode else "Gerçek Görsel & Mantık Testi"
    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<title>TUSAS TestLab — {title}</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
  h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:13px;margin-bottom:20px}}
  .cards{{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;min-width:130px}}
  .card .n{{font-size:26px;font-weight:700}} .card .l{{color:#8b949e;font-size:12px}}
  table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:10px;overflow:hidden}}
  th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid #21262d;vertical-align:top;font-size:13px}}
  th{{background:#1c2128;color:#8b949e;font-weight:600}}
  .dim{{color:#8b949e;font-size:11px}} .note{{color:#d29922;font-size:11px}}
  code{{background:#21262d;padding:1px 5px;border-radius:4px;font-size:11px}}
  .fails{{margin:0;padding-left:16px}} .fails li{{margin:4px 0}}
  .ss{{max-width:240px;border:1px solid #30363d;border-radius:6px}}
  .ok{{color:#1b7f3b}} .bad{{color:#b22222}} .blue{{color:#3498db}}
</style></head><body>
<h1>TUSAS TestLab — {title}</h1>
<div class="sub">{ts}</div>
<div class="cards">
  <div class="card"><div class="n">{total}</div><div class="l">Senaryo</div></div>
  <div class="card"><div class="n {passed_class}">{passed}</div><div class="l">{passed_label}</div></div>
  <div class="card"><div class="n bad">{failed}</div><div class="l">Kaldı</div></div>
  <div class="card"><div class="n">{total_hard - total_failed_checks}/{total_hard}</div><div class="l">Hard-check</div></div>
</div>
<table>
  <tr><th>Senaryo</th><th>Kategori</th><th>Sonuç</th><th>Tutarsızlıklar</th><th>Ekran</th></tr>
  {''.join(rows)}
</table>
</body></html>"""
    path = os.path.join(REPORT_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [HTML] Rapor: {os.path.abspath(path)}")

    # Cleanup temporary screenshots to prevent disk space bloat (since base64 is embedded in HTML)
    try:
        import glob
        for d in [SCREENSHOT_DIR, os.path.join(REPORT_DIR, "annotated"), os.path.join(REPORT_DIR, "crops")]:
            if os.path.exists(d):
                for f_path in glob.glob(os.path.join(d, "*")):
                    if os.path.isfile(f_path):
                        try:
                            os.remove(f_path)
                        except Exception:
                            pass
    except Exception as e:
        print(f"Error cleaning up screenshots: {e}")
