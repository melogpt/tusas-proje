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
from typing import List

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



def _inject_known_render_fault(pencere, scenario, idx: int) -> str:
    """
    Dedektör öz-testi için, dürüst kontrollerin YAKALAMASI gereken, açıkça
    tanımlı bir render hatası enjekte eder. Her hata tipi belirli bir kontrole
    eşlenir; böylece testin boş olmadığı kanıtlanır:
      • wrong_number → model_value
      • wrong_unit   → model_value (birim)
      • wrong_fill   → bar_fill_render
      • false_valid  → model_value (INVALID parametrede sayı göstermek)
      • bad_anti     → anti_ice_valid (inject'siz senaryolar için)
    Döndürür: enjekte edilen hata tipinin adı.
    """
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

    if target_key is None:
        pencere.lbl_anti.setText("ERR")
        return "ANTI_ICE (bad_anti)"

    spec = getattr(w, "spec", w)
    available_fts = []
    if target_key in getattr(scenario, "invalid_params", []):
        available_fts.append("false_valid")
    else:
        available_fts.append("wrong_number")
        if hasattr(w, "lbl_val"):
            available_fts.append("wrong_subtle")
            cur = w.lbl_val.text()
            if "C" in cur or "%" in cur or "PSI" in cur:
                available_fts.append("wrong_unit")
        if hasattr(w, "bar"):
            if getattr(spec, "orientation", "V").upper() == "V":
                available_fts.append("wrong_fill")
            available_fts.append("wrong_color")

    import random
    ft = random.choice(available_fts)

    if ft == "wrong_number":
        vmax = getattr(spec, "vmax", None)
        bad = (vmax + 999) if vmax is not None else 99999
        if hasattr(w, "lbl_val"):
            w.lbl_val.setText(f"{bad:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())
        elif hasattr(w, "value"):
            w.value = bad
            w.update()
    elif ft == "wrong_unit":
        if hasattr(w, "lbl_val"):
            cur = w.lbl_val.text()
            if "C" in cur:
                w.lbl_val.setText(cur.replace("C", "F"))
            elif "%" in cur:
                w.lbl_val.setText(cur.replace("%", "KG"))
            elif "PSI" in cur:
                w.lbl_val.setText(cur.replace("PSI", "BAR"))
            else:
                w.lbl_val.setText((cur.split(" ")[0] if " " in cur else cur) + " XXX")
    elif ft == "wrong_fill":
        if hasattr(w, "bar"):
            vmin = getattr(spec, "vmin", 0)
            vmax = getattr(spec, "vmax", 100)
            cur = pencere.vals.get(target_key, vmin)
            w.bar._value = vmin if cur > (vmin + vmax) / 2 else vmax
            w.bar.update()
        else:
            ft = "wrong_number"
    elif ft == "wrong_color":
        if hasattr(w, "bar"):
            w.bar._intended_color_hack = w.bar._color
            from PyQt5.QtGui import QColor
            w.bar._color = QColor("#800080")
            w.bar.update()
    elif ft == "wrong_subtle":
        if hasattr(w, "lbl_val"):
            cur = pencere.vals.get(target_key, 0)
            w.lbl_val.setText(f"{cur + 0.2:.1f}")
    elif ft == "false_valid":
        if hasattr(w, "set_invalid"):
            w.set_invalid(False)
        if hasattr(w, "lbl_val"):
            vmin = getattr(spec, "vmin", 0)
            w.lbl_val.setText(f"{vmin:.{getattr(spec, 'decimals', 1)}f} {getattr(spec, 'unit', '')}".strip())

    return f"{target_key} ({ft})"



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
    injected_fault: str = ""


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
    w.show()
    w.resize(1600, 860)
    QApplication.processEvents()
    yield w
    w.close()
    QApplication.processEvents()


# ─── ANA TEST ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.id for s in ALL_SCENARIOS])
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

    # Anti-ice: geçerli bir durum (OFF/AUTO/ON) render edilmeli
    anti_txt = pencere.lbl_anti.text().strip().upper()
    anti_ok = anti_txt in VALID_ANTI

    # 5) Hard-check'leri topla
    hard, failed, checks_dump = [], [], []
    for rep in reports:
        for c in rep.hard_checks:
            hard.append(c)
            if not c.passed:
                nom_crop = _get_cropped_b64(nominal_pil, c.bbox)
                act_crop = _get_cropped_b64(pil, c.bbox)
                failed.append({
                    "name": c.name,
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                    "note": c.note,
                    "nominal_crop_b64": nom_crop,
                    "actual_crop_b64": act_crop,
                    "param": rep.key
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
            "param": "ANTI"
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
        injected_fault=injected_fault or "",
    )

    if inject_mode:
        # Dedektör, enjekte edilen hatayı YAKALAMALI (en az bir hard-check FAIL)
        caught = len(failed) > 0
        res.detector_caught = caught
        res.passed = caught
        _results.append(res)
        assert caught, (
            f"[DEDEKTÖR ZAYIF] {scenario.id}: bilinen render hatası enjekte edildi "
            f"ama hiçbir hard-check yakalamadı. Test gerçek hatayı kaçırıyor olabilir."
        )
    else:
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
        } for r in _results], f, ensure_ascii=False, indent=2)


def _img_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _save_html():
    total = len(_results)
    passed = sum(1 for r in _results if r.passed)
    failed = total - passed
    total_hard = sum(r.n_hard for r in _results)
    total_failed_checks = sum(r.n_failed for r in _results)
    detector_mode = any(r.detector_caught is not None for r in _results)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    global_seen_failures = set()
    global_seen_misses = set()
    for r in _results:
        if detector_mode and not r.passed and r.injected_fault:
            if r.injected_fault in global_seen_misses:
                continue
            global_seen_misses.add(r.injected_fault)

        if detector_mode and r.passed:
            badge = "YAKALANDI"
            color = "#3498db"
        else:
            badge = "PASS" if r.passed else "FAIL"
            color = "#1b7f3b" if r.passed else "#b22222"
        det = ""
        if r.detector_caught is not None:
            if r.detector_caught:
                det = "✓ hata yakalandı"
            else:
                det = "✗ hata KAÇIRILDI"
                if r.injected_fault:
                    det += f"<br><span style='color:#d29922;font-size:11px;margin-top:4px;display:inline-block;'>Enjekte edilen:<br><b>{r.injected_fault}</b></span>"
        fail_html = ""
        if r.failures:
            items = []
            for f in r.failures:
                fail_key = (f['name'], f['expected'], f['actual'])
                if fail_key in global_seen_failures:
                    continue
                global_seen_failures.add(fail_key)
                
                item_html = (
                    f"<li><b>{f['name']}</b>: beklenen <code>{f['expected']}</code> | "
                    f"gerçek <code>{f['actual']}</code>"
                )
                if f.get('note'):
                    item_html += f"<br><span class='note'>{f['note']}</span>"
                
                # Add side-by-side cropped images
                if f.get('nominal_crop_b64') or f.get('actual_crop_b64'):
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
                    item_html += "</div>"
                
                item_html += "</li>"
                items.append(item_html)
            
            # If this scenario has NO NEW ERRORS (they were all seen before), skip rendering the whole row to avoid clutter!
            if not items and detector_mode and r.passed:
                continue
                
            fail_html = f"<ul class='fails'>{''.join(items)}</ul>"
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
    print(f"\n  🌐 Rapor: {os.path.abspath(path)}")
