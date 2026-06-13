"""
TUSAS TestLab — Deterministik Test Düzeneği (Harness)
======================================================
Uygulamanın _tick_sim'i her tikte TÜM değerleri rastgele oynatır, düşük
olasılıkla parametreleri INVALID yapar ve WCA deposunu biriktirir. Ayrıca
_sim_init kalıcı "DEMO WARNING/CAUTION" girdileri ekler.

Dürüst ve TEKRARLANABİLİR test için bu modül:
  • random modülünü geçici olarak gürültüsüz/deterministik yapar
    (random.random→1.0, random.uniform→orta nokta) — böylece sadece enjekte
    edilen arıza eşik geçer, başka parametre rastgele alarm vermez.
  • Sadece senaryonun enjekte ettiği değerleri + temiz nominal tabanı kullanır.
  • WCA'nın gerçek arızadan dolması için kontrollü sayıda tick atar.
  • Kalıcı DEMO girdilerini ayıklamak için yardımcı sağlar.

UI dosyalarına (ekran.py, widgets.py ...) DOKUNULMAZ — sadece çalışma anında
random davranışı sarmalanır ve geri alınır.
"""

from __future__ import annotations

import random
import time
from typing import List

from PyQt5.QtWidgets import QApplication

# Kalıcı (tasarım gereği her zaman var olan) demo WCA anahtarları.
DEMO_WCA_KEYS = {"WRN_DEMO", "CAU_DEMO"}
ADVISORY_PREFIXES = ("ADV_",)


class _DeterministicRandom:
    """random modülünü geçici olarak gürültüsüz yapan context manager."""
    def __init__(self, seed: int = 1337):
        self.seed = seed
        self._saved = {}

    def __enter__(self):
        for name in ("random", "uniform", "randint", "choice", "gauss", "normalvariate"):
            self._saved[name] = getattr(random, name, None)
        random.seed(self.seed)
        random.random = lambda: 1.0                      # düşük olasılıklı olaylar tetiklenmez
        random.uniform = lambda a, b: (a + b) / 2.0      # nudge gürültüsü 0 etrafında
        random.randint = lambda a, b: a
        random.gauss = lambda mu, sigma: mu
        random.normalvariate = lambda mu, sigma: mu
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            if fn is not None:
                setattr(random, name, fn)


class _LockedDict(dict):
    """
    Belirli anahtarları kilitleyen dict alt sınıfı. _tick_sim türetilmiş
    parametreleri (örn. E2_TIT = 480 + E2_TRQ*3) her tikte yeniden hesaplayıp
    enjekte ettiğimiz değeri EZER. Bu sözlük, kilitli anahtarlara yazmayı
    yok sayar → enjekte edilen arıza değeri tik boyunca SABİT kalır ve
    uygulamanın GERÇEK WCA/FaultGate hattı bu değerle çalışır.
    UI dosyalarına dokunmadan, sadece veri kabını sarmalıyoruz.
    """
    def __init__(self, base: dict, locked: dict):
        super().__init__(base)
        self._locked = dict(locked)
        for k, v in self._locked.items():
            super().__setitem__(k, v)

    def __setitem__(self, key, value):
        if key in self._locked:
            return                       # kilitli → enjekte değeri koru
        super().__setitem__(key, value)


def deterministic_apply(pencere, scenario, ticks: int = 2) -> None:
    """
    Senaryoyu UI'a deterministik biçimde uygular ve WCA'yı GERÇEK arızadan
    doldurur. Rastgele alarm/INVALID üretmez (enjekte edilenler hariç).

    Önemli: enjekte edilen değerler türetilmiş hesaplarla ezilmesin diye
    pencere.vals / pencere.invalid kilitli sözlükle sarmalanır; böylece
    uygulamanın kendi _tick_sim → FaultGate → WcaStore hattı, doğru değerle
    ve gerçek debounce süresiyle çalışıp WCA'yı oluşturur.
    """
    with _DeterministicRandom(seed=1337):
        pencere.t_sim.stop()

        # INVALID bayraklarını ayarla (sadece senaryonun istediği True)
        for key in list(pencere.invalid.keys()):
            pencere.invalid[key] = key in scenario.invalid_params

        # Enjekte değerleri ve invalid bayraklarını kilitle
        locked_vals = {k: float(v) for k, v in scenario.inject.items()}
        locked_inv = {k: True for k in scenario.invalid_params}
        pencere.vals = _LockedDict(pencere.vals, locked_vals)
        pencere.invalid = _LockedDict(pencere.invalid, locked_inv)

        # 1) İlk tik: FaultGate kapıları AKTİF olur (start_ms = şimdi)
        pencere._tick_sim()

        # 2) Debounce süresi GERÇEKTEN geçsin diye zamanı ilerlet
        secs = max(scenario.time_advance_secs, 12)
        pencere.elapsed = pencere.elapsed.addSecs(secs)

        # 3) Sonraki tikler: (now - start_ms) >= min_*_s → WCA'ya yazılır
        for _ in range(max(ticks, 1)):
            pencere._tick_sim()

        pencere._apply_ui()
        now = pencere._now_ms()
        pencere._render_wca(now)
        QApplication.processEvents()
        time.sleep(0.05)


def real_wca_entries(pencere) -> List:
    """DEMO ve advisory dışındaki GERÇEK (arıza kaynaklı) WCA girdileri."""
    out = []
    for e in pencere.wca.snapshot_sorted():
        if e.key in DEMO_WCA_KEYS:
            continue
        if any(e.key.startswith(p) for p in ADVISORY_PREFIXES):
            continue
        out.append(e)
    return out


def measure_fill_brightness(crop_rgb) -> float:
    """
    Dikey bar doluluk oranını PARLAKLIK ile ölç (0..1).
    Fill renkleri parlaktır (#FF3333, #00FF66 ...). Warning/Caution arka fonu
    (#3A0000/#3A2600) ve normal fon (#050505) koyudur → dolu sayılmaz.
    Beyaz border da elenir. Bu, eski measure_bar_fill_ratio'nun renkli arka
    fonları yanlışlıkla "dolu" sayma hatasını giderir.
    """
    import numpy as np
    arr = np.asarray(crop_rgb)
    if arr.ndim < 3:
        return -1.0
    h, w = arr.shape[:2]
    if h < 6 or w < 6:
        return -1.0
    inner = arr[2:h - 2, 2:w - 2].astype(float)
    if inner.shape[0] == 0:
        return -1.0
    r, g, b = inner[:, :, 0], inner[:, :, 1], inner[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    is_white = (r > 200) & (g > 200) & (b > 200)
    is_bright_fill = (mx > 120) & (~is_white)     # parlak ama beyaz değil → fill
    row_ratio = np.mean(is_bright_fill, axis=1)
    filled_rows = int(np.sum(row_ratio > 0.30))
    return filled_rows / inner.shape[0]
