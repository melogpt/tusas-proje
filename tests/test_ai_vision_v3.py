"""
TUSAŞ TestLab — AI Vision Test Runner
======================================
Her senaryo için:
  1. Hata parametrelerini enjekte eder (FaultGate bypass ile)
  2. MC / WCA logic testini yapar
  3. Ekranın screenshot'ını alır  →  test_reports/screenshots/<ID>.png
  4. Claude Vision API'ye gönderir (--no-ai ile atlanır)
  5. Sonuçları JSON + HTML rapora yazar  →  test_reports/report.html

KULLANIM:
  Tam test (AI Vision dahil, ~90 sn):
      python -m pytest tests/test_ai_vision.py -v -s

  Hızlı test (sadece logic + screenshot, ~15 sn):
      python -m pytest tests/test_ai_vision.py -v -s --no-ai

  Tek senaryo:
      python -m pytest tests/test_ai_vision.py -v -s -k "ENG_001"

  Raporu aç (testler bittikten sonra):
      start test_reports\\report.html
"""

import base64
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytest
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ekran import FlightDisplay
from tests.fault_scenarios import SCENARIOS, FaultScenario

# Lokal analiz modülleri (opsiyonel — kurulu değilse atlanır)
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


# ─── --no-ai SEÇENEĞI ────────────────────────────────────────────────────────
# pytest_addoption conftest.py'de tanımlı

@pytest.fixture(scope="session")
def use_ai(request):
    return not request.config.getoption("--no-ai")

# ─── RAPOR KLASÖRÜ ────────────────────────────────────────────────────────────

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_reports")
os.makedirs(REPORT_DIR, exist_ok=True)
SCREENSHOT_DIR = os.path.join(REPORT_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─── SONUÇ DATACLASS ─────────────────────────────────────────────────────────

@dataclass
class VisionTestResult:
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    timestamp: str
    # Logic test sonuçları
    mc_state_ok: bool = False
    wca_text_ok: bool = False
    # AI Vision sonuçları
    ai_response: str = ""
    ai_pass: bool = False
    ai_confidence: str = "UNKNOWN"   # "HIGH" | "MEDIUM" | "LOW" | "ERROR"
    # ML model sonuçları
    ml_prediction: str = ""          # "WARNING / %97" gibi
    ml_dataset_count: int = 0        # toplam toplanan örnek sayısı
    # Genel
    overall_pass: bool = False
    screenshot_path: str = ""
    error: str = ""
    duration_ms: int = 0


# ─── SCREENSHOT YARDIMCISI ────────────────────────────────────────────────────

def take_screenshot(widget, scenario_id: str = "") -> Tuple[QPixmap, str]:
    """Widget'ın screenshot'ını al. Senaryo başına TEK kez çağrılır."""
    widget.repaint()
    QApplication.processEvents()
    time.sleep(0.15)

    pixmap = widget.grab()
    name = scenario_id if scenario_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    pixmap.save(path, "PNG")

    # base64'e çevir
    buf = QByteArray()
    qbuf = QBuffer(buf)
    qbuf.open(QIODevice.WriteOnly)
    pixmap.save(qbuf, "PNG")
    b64 = base64.b64encode(buf.data()).decode("utf-8")

    return pixmap, path, b64


# ─── CLAUDE VISION API ────────────────────────────────────────────────────────

def call_claude_vision(image_b64: str, prompt: str) -> Tuple[str, str]:
    """
    Claude Vision API'yi çağır.
    Döndürür: (yanıt_metni, güven_seviyesi)
    API anahtarı otomatik ekleniyor (proxy üzerinden).
    """
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Sen bir aviyonik test mühendisisin. "
                            "Bu ekran görüntüsü bir uçuş sistemleri gösterge panelini gösteriyor.\n\n"
                            f"SORU: {prompt}\n\n"
                            "Yanıtını şu formatta ver:\n"
                            "SONUÇ: [EVET/HAYIR]\n"
                            "GÜVEN: [YÜKSEK/ORTA/DÜŞÜK]\n"
                            "AÇIKLAMA: [Kısa açıklama]\n\n"
                            "Birden fazla soru varsa her biri için ayrı ayrı yanıtla."
                        ),
                    },
                ],
            }
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body["content"][0]["text"]
            # Güven seviyesini parse et
            if "GÜVEN: YÜKSEK" in text:
                confidence = "HIGH"
            elif "GÜVEN: ORTA" in text:
                confidence = "MEDIUM"
            elif "GÜVEN: DÜŞÜK" in text:
                confidence = "LOW"
            else:
                confidence = "UNKNOWN"
            return text, confidence
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return f"API HATASI {e.code}: {error_body}", "ERROR"
    except Exception as ex:
        return f"BAĞLANTI HATASI: {ex}", "ERROR"


def parse_ai_pass(ai_response: str) -> bool:
    """
    AI yanıtını analiz et.
    Birden fazla soru varsa tüm EVET'ler geçmeli.
    """
    if not ai_response or "HATA" in ai_response.upper():
        return False
    lines = ai_response.upper().split("\n")
    sonuc_lines = [l for l in lines if "SONUÇ:" in l or "SONUC:" in l]
    if not sonuc_lines:
        # Genel "EVET" var mı bak
        return "EVET" in ai_response.upper()
    hayir_count = sum(1 for l in sonuc_lines if "HAYIR" in l)
    return hayir_count == 0


# ─── SENARYO RUNNER ───────────────────────────────────────────────────────────

def run_scenario_logic(pencere: FlightDisplay, scenario: FaultScenario) -> Tuple[bool, bool]:
    """
    Senaryoyu uygula, logic testleri yap.
    Döndürür: (mc_state_ok, wca_text_ok)

    FaultGate sorunu: _tick_sim içinde (now_ms - gate.start_ms) farkı hesaplanıyor.
    Hem elapsed hem now_ms birlikte ilerlediği için fark hep ~0 kalır.
    Çözüm: elapsed'ı büyük bir değere set edip, ardından FaultGate'deki
    tüm mevcut state'lerin start_ms'ini 0'a çekerek farkı garantilemek.
    """
    # Parametreleri enjekte et
    for key, val in scenario.inject.items():
        pencere.vals[key] = val

    # Invalid parametreleri işaretle
    for key in scenario.invalid_params:
        pencere.invalid[key] = True

    # elapsed'ı yeterince ilerlet (min_warning_s = 5, min_caution_s = 4)
    pencere.elapsed = pencere.elapsed.addSecs(max(scenario.time_advance_secs, 10))

    # İlk tick — FaultGate'e state'leri kaydettirir (start_ms = now_ms)
    for key, val in scenario.inject.items():
        pencere.vals[key] = val
    pencere._tick_sim()

    # FaultGate'deki tüm aktif state'lerin start_ms'ini 0'a çek
    # → bir sonraki tick'te (now_ms - 0) >> min_warning_s olur, gate açılır
    for st in pencere.fgate._s.values():
        if st.active:
            st.start_ms = 0

    # Sonraki tick(ler) — gate açık, ama _tick_sim bazı parametrelerin
    # üzerine yazıyor (örn. ENV_CABALT FLT_ALT'tan hesaplanıyor).
    # Çözüm: _tick_sim'i wrap edip her tick SONRASI inject'i yeniden uygula.
    original_tick = pencere._tick_sim.__func__

    def _patched_tick(self):
        original_tick(self)
        # _tick_sim'in üzerine yazdığı parametreleri geri yükle
        for key, val in scenario.inject.items():
            self.vals[key] = val

    import types
    pencere._tick_sim = types.MethodType(_patched_tick, pencere)

    for _ in range(max(scenario.tick_count, 2)):
        pencere._tick_sim()

    # Patch'i geri al
    pencere._tick_sim = types.MethodType(original_tick, pencere)

    QApplication.processEvents()
    time.sleep(0.1)

    # ── Master Caution kontrolü ──────────────────────────────────────────────
    mc_text = pencere.lbl_mc.text()
    mc_style = pencere.lbl_mc.styleSheet().upper()

    if scenario.expected_mc_state == "WARNING":
        mc_ok = mc_text == "ON" and "#FF3333" in mc_style
    elif scenario.expected_mc_state == "CAUTION":
        mc_ok = mc_text == "ON" and ("#FFB000" in mc_style or "#FF3333" in mc_style)
    else:  # OFF (NOM_001)
        mc_ok = mc_text == "OFF"

    # ── WCA metin kontrolü ───────────────────────────────────────────────────
    wca_entries = pencere.wca.snapshot_sorted()
    all_texts = " ".join(e.text.upper() for e in wca_entries)

    if scenario.expected_wca_texts:
        wca_ok = any(
            expected.upper() in all_texts
            for expected in scenario.expected_wca_texts
        )
    else:
        wca_ok = True   # beklenen metin tanımlanmamışsa geç

    return mc_ok, wca_ok


# ─── PYTEST FIXTURE ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """QApplication (modül bazında paylaşılır)."""
    a = QApplication.instance() or QApplication(sys.argv)
    yield a


@pytest.fixture
def pencere(app, monkeypatch):
    """Ses sistemi devre dışı bırakılmış FlightDisplay."""
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)
    w = FlightDisplay()
    w.show()
    w.resize(1600, 860)
    QApplication.processEvents()
    yield w
    w.close()


# ─── TEST SONUÇ TOPLAMA ───────────────────────────────────────────────────────

_all_results: List[VisionTestResult] = []


def _save_results(open_browser: bool = False):
    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(REPORT_DIR, "vision_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in _all_results], f, ensure_ascii=False, indent=2)

    # ── HTML rapor ────────────────────────────────────────────────────────────
    _generate_html_report()

    # ── Tarayıcıyı aç (session sonu veya zorla) ───────────────────────────────
    if open_browser:
        html_path = os.path.abspath(os.path.join(REPORT_DIR, "report.html"))
        try:
            import webbrowser
            webbrowser.open(f"file://{html_path}")
        except Exception:
            try:
                import subprocess, sys as _sys
                if _sys.platform == "win32":
                    os.startfile(html_path)
                elif _sys.platform == "darwin":
                    subprocess.Popen(["open", html_path])
                else:
                    subprocess.Popen(["xdg-open", html_path])
            except Exception:
                print(f"  Raporu manuel açın: {html_path}")


def _generate_html_report():
    """Sonuçları + screenshot'ları gösteren tek sayfalık HTML rapor üretir."""
    html_path = os.path.join(REPORT_DIR, "report.html")

    total   = len(_all_results)
    passed  = sum(1 for r in _all_results if r.overall_pass)
    failed  = total - passed
    ai_ok   = sum(1 for r in _all_results if r.ai_pass)
    pct     = int(passed / total * 100) if total else 0

    rows_html = ""
    for r in _all_results:
        # screenshot'u base64'e göm (dosya varsa)
        img_tag = "<span style='color:#666'>screenshot yok</span>"
        if r.screenshot_path and os.path.exists(r.screenshot_path):
            with open(r.screenshot_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:4px;cursor:pointer" '
                f'onclick="openImg(this)" />' 
            )

        status_bg  = "#1a3a1a" if r.overall_pass else "#3a1a1a"
        status_txt = "#4caf50" if r.overall_pass else "#f44336"
        status_lbl = "PASS" if r.overall_pass else "FAIL"
        mc_ok  = "✅" if r.mc_state_ok  else "❌"
        wca_ok = "✅" if r.wca_text_ok  else "❌"
        ai_lbl = ("✅ PASS" if r.ai_pass else
                  ("❌ FAIL" if r.ai_confidence not in ("ERROR","UNKNOWN","") else "⏭ SKIP"))
        ai_resp = (r.ai_response[:300] + "…") if len(r.ai_response) > 300 else r.ai_response
        ai_resp = ai_resp.replace("<","&lt;").replace(">","&gt;")
        ml_pred = r.ml_prediction if r.ml_prediction else f"Veri toplanıyor ({r.ml_dataset_count}/30)" 

        rows_html += f"""
        <div class="card" style="border-left:4px solid {status_txt}; background:{status_bg}">
          <div class="card-header">
            <div>
              <span class="sid">[{r.scenario_id}]</span>
              <span class="sname">{r.scenario_name}</span>
              <span class="cat-badge cat-{r.category}">{r.category}</span>
              <span class="sev-badge sev-{r.severity}">{r.severity[:4]}</span>
            </div>
            <span class="overall" style="color:{status_txt}">{status_lbl}</span>
          </div>
          <div class="card-body">
            <div class="screenshot-col">{img_tag}</div>
            <div class="info-col">
              <div class="checks">
                <div>{mc_ok} <b>Master Caution</b></div>
                <div>{wca_ok} <b>WCA Metin</b></div>
                <div>{ai_lbl} <b>AI Vision</b></div>
                <div>&#129302; <b>ML:</b> <span style="color:#2196f3">{ml_pred}</span></div>
                <div style="color:#888;font-size:11px">{r.duration_ms} ms</div>
              </div>
              <div class="ai-box">{ai_resp if ai_resp else "<i style='color:#555'>AI analizi yapılmadı</i>"}</div>
            </div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>TUSAŞ TestLab — AI Vision Raporu</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: Consolas, monospace; font-size: 13px; }}
  .topbar {{ background: #111; border-bottom: 1px solid #333; padding: 16px 24px;
             display: flex; justify-content: space-between; align-items: center; }}
  .topbar h1 {{ font-size: 15px; color: #fff; letter-spacing: .05em; }}
  .topbar span {{ font-size: 12px; color: #888; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
              padding: 20px 24px; }}
  .metric {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 6px;
             padding: 14px 18px; }}
  .metric .label {{ font-size: 10px; color: #666; letter-spacing: .08em;
                    text-transform: uppercase; margin-bottom: 6px; }}
  .metric .value {{ font-size: 28px; font-weight: bold; }}
  .metric.pass .value {{ color: #4caf50; }}
  .metric.fail .value {{ color: #f44336; }}
  .metric.ai   .value {{ color: #2196f3; }}
  .metric.pct  .value {{ color: #ff9800; }}
  .cards {{ padding: 0 24px 40px; display: flex; flex-direction: column; gap: 14px; }}
  .card {{ background: #111; border-radius: 8px; border: 1px solid #2a2a2a;
           overflow: hidden; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center;
                  padding: 10px 16px; border-bottom: 1px solid #1e1e1e; }}
  .sid {{ color: #888; margin-right: 8px; }}
  .sname {{ color: #fff; font-weight: bold; margin-right: 10px; }}
  .cat-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 99px;
                background: #222; color: #aaa; border: 1px solid #333; margin-right: 4px; }}
  .sev-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 99px;
                background: #1a1a2e; color: #9fa8da; border: 1px solid #333; }}
  .sev-WARN {{ background: #2a1a1a; color: #f44336; }}
  .sev-CAUT {{ background: #2a2a1a; color: #ff9800; }}
  .sev-ADVI {{ background: #1a2a1a; color: #4caf50; }}
  .overall {{ font-size: 13px; font-weight: bold; letter-spacing: .05em; }}
  .card-body {{ display: grid; grid-template-columns: 420px 1fr; gap: 16px;
                padding: 14px 16px; }}
  .screenshot-col img {{ display: block; }}
  .info-col {{ display: flex; flex-direction: column; gap: 10px; }}
  .checks {{ display: flex; flex-direction: column; gap: 6px; line-height: 1.6; }}
  .ai-box {{ background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 4px;
             padding: 10px; font-size: 11px; color: #aaa; line-height: 1.7;
             white-space: pre-wrap; flex: 1; overflow-y: auto; max-height: 160px; }}
  /* lightbox */
  #lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.92);
         z-index:999; align-items:center; justify-content:center; cursor:zoom-out; }}
  #lb img {{ max-width:92vw; max-height:92vh; border-radius:4px; }}
  #lb.open {{ display:flex; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>TUSAŞ TestLab — AI Vision Test Raporu</h1>
  <span>Oluşturulma: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</span>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Toplam</div><div class="value">{total}</div></div>
  <div class="metric pass"><div class="label">Pass</div><div class="value">{passed}</div></div>
  <div class="metric fail"><div class="label">Fail</div><div class="value">{failed}</div></div>
  <div class="metric ai"><div class="label">AI Vision Pass</div><div class="value">{ai_ok}</div></div>
</div>
<div class="cards">{rows_html}</div>
<div id="lb" onclick="this.classList.remove('open')">
  <img id="lb-img" src="" />
</div>
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


# ─── PARAMETRIK TEST ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_vision_scenario(pencere, scenario: FaultScenario, use_ai):
    """Her senaryo için: logic test + screenshot + AI Vision analizi."""
    t_start = time.time()
    result = VisionTestResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        category=scenario.category,
        severity=scenario.severity,
        timestamp=datetime.now().isoformat(),
    )

    idx = SCENARIOS.index(scenario) + 1
    total_s = len(SCENARIOS)
    print(f"\n[{idx:02d}/{total_s}] {scenario.id}  {scenario.name}")
    print(f"       ├ Kategori: {scenario.category}  Seviye: {scenario.severity}")

    try:
        # 1. Logic testleri
        mc_ok, wca_ok = run_scenario_logic(pencere, scenario)
        result.mc_state_ok = mc_ok
        result.wca_text_ok = wca_ok

        t_logic = int((time.time() - t_start) * 1000)
        mc_sym  = "PASS" if mc_ok  else "FAIL"
        wca_sym = "PASS" if wca_ok else "FAIL"
        print(f"       ├ Logic   MC={mc_sym}  WCA={wca_sym}  ({t_logic}ms)")

        # 2. Screenshot al
        t_ss = time.time()
        _, screenshot_path, b64 = take_screenshot(pencere, scenario.id)
        result.screenshot_path = screenshot_path
        print(f"       ├ Screenshot kaydedildi  ({int((time.time()-t_ss)*1000)}ms)  → {os.path.basename(screenshot_path)}")

        # 3a. OpenCV lokal analiz (offline, ~5ms)
        if HAS_CV_ANALYZER:
            cv_result = analyze_screenshot(open(screenshot_path, "rb").read())
            print_cv_result(cv_result, scenario.id)
            result.ai_response += f"[CV] overall={cv_result.overall_state} wca_red={cv_result.wca_is_red}\n"

        # 3b. ML model tahmini + veri toplama
        if HAS_ML:
            # Veriyi eğitim setine ekle (her test = 6 augmented örnek)
            collect_training_data(screenshot_path, scenario.id, scenario.severity)
            summary = dataset_summary()
            # Model eğitilmişse tahmin yap
            pred = predict(screenshot_path)
            result.ml_dataset_count = summary['total']
            if "error" not in pred:
                conf_pct = pred['confidence'] * 100
                anom = " ⚠️ANOMALY" if pred.get("anomaly") else ""
                result.ml_prediction = f"{pred['class']}  %{conf_pct:.0f}{anom}"
                ml_info = f"[ML] {pred['class']}  %{conf_pct:.0f}  ({pred['ms']}ms){anom}"
                print(f"       ├ {ml_info}")
                result.ai_response += f"\n{ml_info}"
            else:
                result.ml_prediction = f"Model yok — önce: python ml_trainer_v3.py generate && python ml_trainer_v3.py train --fast"
                ml_info = f"[ML] Model bekleniyor — {summary['total']} örnek toplandı"
                print(f"       ├ {ml_info}")
                result.ai_response += f"\n{ml_info}"

        # 3c. Claude Vision API analizi
        if scenario.ai_vision_prompt and use_ai:
            print(f"       ├ Claude Vision API bekleniyor...", end="", flush=True)
            t_ai = time.time()
            ai_response, confidence = call_claude_vision(b64, scenario.ai_vision_prompt)
            result.ai_response = ai_response
            result.ai_confidence = confidence
            result.ai_pass = parse_ai_pass(ai_response)
            ai_sym = "PASS" if result.ai_pass else "FAIL"
            print(f"  {ai_sym} [{confidence}]  ({int((time.time()-t_ai)*1000)}ms)")
        elif not use_ai:
            print(f"       ├ AI Vision: ATLANDI (--no-ai)")

        # 4. Genel sonuç
        result.overall_pass = result.mc_state_ok and result.wca_text_ok
        if scenario.ai_vision_prompt and result.ai_confidence not in ("ERROR", "UNKNOWN"):
            result.overall_pass = result.overall_pass and result.ai_pass

    except Exception as ex:
        result.error = str(ex)
        result.overall_pass = False
        print(f"   💥 HATA: {ex}")

    result.duration_ms = int((time.time() - t_start) * 1000)
    _all_results.append(result)

    final = "PASS" if result.overall_pass else "FAIL"
    bar   = "█" * 20
    print(f"       └ SONUÇ: {final}  (toplam {result.duration_ms}ms)")

    _save_results()

    # Soft fail — test başarısız olsa bile rapor üretilir ve tarayıcı açılır
    if not result.mc_state_ok:
        pytest.fail(
            f"[{scenario.id}] Master Caution beklenen={scenario.expected_mc_state}, "
            f"gerçek={pencere.lbl_mc.text()}",
            pytrace=False,
        )
    if not (result.wca_text_ok or not scenario.expected_wca_texts):
        pytest.fail(
            f"[{scenario.id}] WCA'da beklenen metin bulunamadı: {scenario.expected_wca_texts}",
            pytrace=False,
        )


# ─── SESSION SONUNDA ÖZET ─────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    html_path = os.path.abspath(os.path.join(REPORT_DIR, "report.html"))

    # Sonuç varsa kaydet, yoksa boş rapor oluştur
    if _all_results:
        total     = len(_all_results)
        passed    = sum(1 for r in _all_results if r.overall_pass)
        failed    = total - passed
        ai_passed = sum(1 for r in _all_results if r.ai_pass)
        print(f"\n{'='*55}")
        print(f"  TUSAS TestLab — Test Tamamlandi")
        print(f"  Toplam={total}  PASS={passed}  FAIL={failed}  AI={ai_passed}")
        print(f"  Rapor : {html_path}")
        print(f"{'='*55}")
        # HTML + JSON kaydet
        json_path = os.path.join(REPORT_DIR, "vision_test_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in _all_results], f, ensure_ascii=False, indent=2)
        _generate_html_report()
    else:
        # Sonuç yoksa basit bir "henüz test yok" sayfası yaz
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<title>TUSAS TestLab</title>
<style>body{background:#0d0d0d;color:#e0e0e0;font-family:sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.box{text-align:center}.title{font-size:20px;color:#fff;margin-bottom:12px}
.sub{color:#666;font-size:13px}</style></head><body>
<div class="box">
  <div class="title">🛩 TUSAŞ TestLab</div>
  <div class="sub">Henüz test sonucu yok.<br>
  <code>python -m pytest tests/test_ai_vision_v3.py -v -s --no-ai</code> çalıştırın.</div>
</div></body></html>""")
        print(f"\n  [!] Test sonucu yok — boş rapor oluşturuldu: {html_path}")

    # Dosya kesinlikle var, şimdi aç
    if os.environ.get("TUSAS_NO_OPEN_REPORT") == "1":
        return
    if not os.path.exists(html_path):
        print(f"  [!] Rapor dosyası bulunamadı: {html_path}")
        return

    import webbrowser, platform, subprocess
    opened = False
    try:
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
        print(f"  🌐 Test raporu tarayıcıda açılıyor...")
        opened = True
    except Exception:
        pass

    if not opened:
        try:
            plat = platform.system()
            if plat == "Windows":
                os.startfile(html_path)
            elif plat == "Darwin":
                subprocess.Popen(["open", html_path])
            else:
                subprocess.Popen(["xdg-open", html_path])
            opened = True
        except Exception:
            pass

    if not opened:
        print(f"  Manuel açın: {html_path}")
