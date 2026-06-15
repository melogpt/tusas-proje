from types import SimpleNamespace

from tests import real_checks
from tests import test_real_vision


def test_value_mismatch_crops_value_label(monkeypatch):
    label = SimpleNamespace(_bbox=(10, 20, 30, 40))
    widget = SimpleNamespace(lbl_val=label, bar=SimpleNamespace(_bbox=(1, 2, 3, 4)))
    window = SimpleNamespace(param_widgets={"E1_TRQ": widget}, text_widgets={})

    monkeypatch.setattr(test_real_vision, "get_widget_bbox", lambda widget, root: widget._bbox)

    bbox = test_real_vision._select_failure_crop_bbox(
        window,
        param_key="E1_TRQ",
        check_name="model_value",
        fallback_bbox=(100, 200, 300, 400),
    )

    assert bbox == (10, 20, 30, 40)


def test_bar_mismatch_keeps_full_widget_so_value_label_stays_visible(monkeypatch):
    bar = SimpleNamespace(_bbox=(50, 60, 70, 80))
    widget = SimpleNamespace(lbl_val=SimpleNamespace(_bbox=(10, 20, 30, 40)), bar=bar)
    window = SimpleNamespace(param_widgets={"E1_TRQ": widget}, text_widgets={})

    monkeypatch.setattr(test_real_vision, "get_widget_bbox", lambda widget, root: widget._bbox)

    bbox = test_real_vision._select_failure_crop_bbox(
        window,
        param_key="E1_TRQ",
        check_name="bar_fill_render",
        fallback_bbox=(100, 200, 300, 400),
    )

    assert bbox == (100, 200, 300, 400)


def test_bar_mismatch_explains_that_text_compares_bar_measurement():
    note = test_real_vision._failure_proof_note("bar_fill_render")

    assert "bar" in note.lower()
    assert "sayı" in note.lower()


def test_bar_fill_summary_includes_screen_value_and_fill_ratio():
    text = real_checks._bar_fill_summary(77.3, "%", 0.70)

    assert "77.3 %" in text
    assert "70%" in text
    assert text.index("77.3 %") < text.index("70%")
