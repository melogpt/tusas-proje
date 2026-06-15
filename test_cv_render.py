from __future__ import annotations

import pytest

from ekran import FlightDisplay
from widgets import SpeedGauge
from cv_utils import (
    grab_widget_bgr,
    assert_speed_needle_angle,
    assert_bar_fill_close,
    assert_bar_color_in,
    assert_invalid_cross_present,
    classify_wca_panel_color,
)


def _build_display(qtbot, monkeypatch) -> FlightDisplay:
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)

    pencere = FlightDisplay()
    qtbot.addWidget(pencere)
    pencere.show()
    qtbot.wait(200)
    return pencere


def test_cv_speed_gauge_needle_angle_widget_only(qtbot):
    g = SpeedGauge("IAS", "KT", 0, 220, caution_hi=200, warning_hi=210)
    qtbot.addWidget(g)
    g.show()
    qtbot.wait(100)

    g.set_value(122)
    qtbot.wait(60)

    img = grab_widget_bgr(g)
    assert_speed_needle_angle(img, expected_value=122, tol_deg=8.0)


def test_cv_speed_gauge_warning_color(qtbot):
    g = SpeedGauge("IAS", "KT", 0, 220, caution_hi=200, warning_hi=210)
    qtbot.addWidget(g)
    g.show()
    qtbot.wait(100)

    g.set_value(215)
    qtbot.wait(60)

    img = grab_widget_bgr(g)

    # Needle angle doğrulamasına ek olarak kırmızı bölge var mı diye kaba kontrol
    # Burada direkt utility yerine daha basit renk beklentisi kullanıyoruz
    from cv_utils import dominant_named_color
    color_name, scores = dominant_named_color(img)

    assert scores["red"] > 100, f"Warning durumunda gauge üzerinde yeterli kırmızı görünmedi: {scores}"


def test_cv_fuel_left_fill_ratio(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    # FUEL_L: 0..600
    w = pencere.param_widgets["FUEL_L"]
    w.set_invalid(False)
    w.set_value(300)  # %50
    qtbot.wait(60)

    img = grab_widget_bgr(w.bar)
    assert_bar_fill_close(img, expected_ratio=0.50, tol=0.12)


def test_cv_fuel_pressure_warning_color(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    # FUEL_P: 0..60, warning_lo=10
    w = pencere.param_widgets["FUEL_P"]
    w.set_invalid(False)
    w.set_value(5)
    qtbot.wait(60)

    img = grab_widget_bgr(w.bar)
    assert_bar_color_in(img, {"red"})


def test_cv_env_smoke_caution_or_warning_color(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    # ENV_SMK: caution_hi=30, warning_hi=50
    w = pencere.param_widgets["ENV_SMK"]
    w.set_invalid(False)
    w.set_value(35)  # caution bölgesi
    qtbot.wait(60)

    img = grab_widget_bgr(w.bar)
    # HSV aralığına göre sarı/turuncu sınıfı dönebilir
    assert_bar_color_in(img, {"yellow", "orange"})


def test_cv_invalid_cross_present(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    w = pencere.param_widgets["FLT_IAS"]
    w.set_invalid(True)
    qtbot.wait(60)

    img = grab_widget_bgr(w.bar)
    assert_invalid_cross_present(img)


def test_cv_wca_warning_panel_is_red(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    # başlangıçta demo warning zaten yükleniyor
    qtbot.wait(60)

    img = grab_widget_bgr(pencere.wca_frame)
    got = classify_wca_panel_color(img)
    assert got == "red", f"WCA warning durumunda kırmızı bekleniyordu, bulunan={got}"


def test_cv_widget_capture_is_stable_after_manual_update(qtbot, monkeypatch):
    pencere = _build_display(qtbot, monkeypatch)

    # IAS küçük bar ve büyük gauge aynı değeri temsil etmeli
    pencere.param_widgets["FLT_IAS"].set_invalid(False)
    pencere.param_widgets["FLT_IAS"].set_value(180)
    pencere.gauge_ias.set_value(180)
    qtbot.wait(80)

    gauge_img = grab_widget_bgr(pencere.gauge_ias)
    small_bar_img = grab_widget_bgr(pencere.param_widgets["FLT_IAS"].bar)

    assert_speed_needle_angle(gauge_img, expected_value=180, tol_deg=8.0)
    assert_bar_fill_close(small_bar_img, expected_ratio=180 / 220, tol=0.15)