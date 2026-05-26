import math
from typing import Optional

from PyQt5.QtCore import Qt, QPoint, QPointF, QRect
from PyQt5.QtGui import QPainter, QPen, QFont, QColor, QCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QDialog
)

from utils import clamp, make_label
from models import ParamSpec


class BarVisual(QWidget):
    def __init__(self, spec: ParamSpec, big: bool = False):
        super().__init__()
        self.spec = spec
        self.big = big
        self._value = spec.vmin
        self._color = QColor("#00FF66")
        self._bg = QColor("#050505")
        self._invalid = False

        if spec.orientation.upper() == "V":
            self.setFixedWidth(34 if big else 28)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setMinimumHeight(60 if big else 30)
        else:
            self.setFixedHeight(16 if big else 12)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_invalid(self, inv: bool):
        self._invalid = bool(inv)
        self.update()

    def set_value(self, v: float):
        self._value = v
        self.update()

    def set_color(self, c: QColor):
        self._color = c
        self.update()

    def set_bg(self, c: QColor):
        self._bg = c
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), self._bg)

        pen = QPen(QColor("#FFFFFF"))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if not self._invalid:
            s = self.spec
            ratio = 0.0 if s.vmax == s.vmin else (self._value - s.vmin) / (s.vmax - s.vmin)
            ratio = clamp(ratio, 0.0, 1.0)
            p.setPen(Qt.NoPen)
            p.setBrush(self._color)
            if s.orientation.upper() == "V":
                h = self.height()
                fill_h = int((h - 4) * ratio)
                r = QRect(2, h - 2 - fill_h, self.width() - 4, fill_h)
            else:
                w = self.width()
                fill_w = int((w - 4) * ratio)
                r = QRect(2, 2, fill_w, self.height() - 4)
            p.drawRect(r)

        if self._invalid:
            pen = QPen(QColor("#FF3333"))
            pen.setWidth(3 if self.big else 2)
            p.setPen(pen)
            p.drawLine(3, 3, self.width() - 4, self.height() - 4)
            p.drawLine(3, self.height() - 4, self.width() - 4, 3)


class ParamHoverPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip)
        self.setWindowFlags(Qt.ToolTip)
        self.setStyleSheet("background-color:#050505; border:1px solid #888;")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.title = make_label("", align=Qt.AlignCenter, size=12, bold=True)
        root.addWidget(self.title)

        self.holder = QWidget()
        self.hlay = QVBoxLayout(self.holder)
        self.hlay.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.holder)

        self.meta = make_label("", align=Qt.AlignLeft, size=10, color="#CCCCCC")
        self.meta.setWordWrap(True)
        root.addWidget(self.meta)

        self._bar: Optional["ParamBar"] = None

    def set_param(self, spec: ParamSpec):
        self.title.setText(spec.label)
        if self._bar is not None:
            self._bar.setParent(None)
            self._bar.deleteLater()
        self._bar = ParamBar(spec, big=True)
        self.hlay.addWidget(self._bar, alignment=Qt.AlignCenter)

    def update_value_and_meta(self, value: float, invalid: bool):
        if not self._bar:
            return
        self._bar.set_invalid(invalid)
        self._bar.set_value(value)
        s = self._bar.spec
        state = self._bar.get_state(value, invalid)

        def fmt(x):
            return "-" if x is None else f"{x:.{s.decimals}f}"

        self.meta.setText(
            f"STATE: {state}\n"
            f"RANGE: {s.vmin:.{s.decimals}f} .. {s.vmax:.{s.decimals}f} {s.unit}\n"
            f"CAUTION: LOW {fmt(s.caution_lo)} | HIGH {fmt(s.caution_hi)}\n"
            f"WARNING: LOW {fmt(s.warning_lo)} | HIGH {fmt(s.warning_hi)}"
        )


class ParamBar(QWidget):
    def __init__(self, spec: ParamSpec, big: bool = False):
        super().__init__()
        self.spec = spec
        self.big = big
        self._invalid = False
        self._last_real = spec.vmin
        self._popup: Optional[ParamHoverPopup] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4 if big else 3)

        self.lbl_name = make_label(spec.label, align=Qt.AlignCenter, size=11 if big else 10, bold=True)
        self.bar = BarVisual(spec, big=big)
        self.lbl_val = make_label("-", align=Qt.AlignCenter, size=11 if big else 10)

        root.addWidget(self.bar, alignment=Qt.AlignHCenter)
        root.addWidget(self.lbl_val)
        root.addWidget(self.lbl_name)

        self.setCursor(Qt.PointingHandCursor)

    def _color_for_value(self, v: float) -> QColor:
        s = self.spec
        if (s.warning_hi is not None and v >= s.warning_hi) or (s.warning_lo is not None and v <= s.warning_lo):
            return QColor("#FF3333")
        if (s.caution_hi is not None and v >= s.caution_hi) or (s.caution_lo is not None and v <= s.caution_lo):
            return QColor("#FFB000")
        ratio = 0.0 if s.vmax == s.vmin else (v - s.vmin) / (s.vmax - s.vmin)
        if ratio < 0.20:
            return QColor("#3399FF")
        if ratio < 0.45:
            return QColor("#00FF66")
        if ratio < 0.75:
            return QColor("#FFFF33")
        if ratio < 0.90:
            return QColor("#FF9933")
        return QColor("#FF3333")

    def _apply_label_bg(self, state: str):
        if state in ("WARNING", "INVALID"):
            bg, fg = "#FF3333", "#000000"
        elif state == "CAUTION":
            bg, fg = "#FFB000", "#000000"
        else:
            bg, fg = None, "#FFFFFF"

        bg_css = f"background-color:{bg}; border-radius:3px; padding:2px 6px;" if bg else "padding:0px;"
        self.lbl_name.setStyleSheet(
            f"color:{fg}; font-size:{11 if self.big else 10}px; font-weight:bold; "
            f"font-family:Consolas, monospace; {bg_css}"
        )

    def _apply_bar_bg(self, state: str):
        if state in ("WARNING", "INVALID"):
            self.bar.set_bg(QColor("#3A0000"))
        elif state == "CAUTION":
            self.bar.set_bg(QColor("#3A2600"))
        else:
            self.bar.set_bg(QColor("#050505"))

    def set_invalid(self, inv: bool):
        self._invalid = bool(inv)
        self.bar.set_invalid(self._invalid)
        if self._invalid:
            self.lbl_val.setText("-")
            self._apply_label_bg("INVALID")
            self._apply_bar_bg("INVALID")
        self._update_popup()

    def set_value(self, v: float):
        s = self.spec
        real = clamp(v, s.vmin, s.vmax)
        self._last_real = real
        if self._invalid:
            self.lbl_val.setText("-")
            self.bar.set_value(s.vmin)
            self.bar.set_color(QColor("#050505"))
            self._apply_label_bg("INVALID")
            self._apply_bar_bg("INVALID")
            self._update_popup()
            return

        self.lbl_val.setText(f"{real:.{s.decimals}f} {s.unit}".strip())
        self.bar.set_value(real)
        self.bar.set_color(self._color_for_value(real))
        st = self.get_state(real, False)
        self._apply_label_bg(st)
        self._apply_bar_bg(st)
        self._update_popup()

    def get_state(self, v: float, invalid: bool = False) -> str:
        if invalid:
            return "INVALID"
        s = self.spec
        if (s.warning_hi is not None and v >= s.warning_hi) or (s.warning_lo is not None and v <= s.warning_lo):
            return "WARNING"
        if (s.caution_hi is not None and v >= s.caution_hi) or (s.caution_lo is not None and v <= s.caution_lo):
            return "CAUTION"
        return "NOMINAL"

    def _update_popup(self):
        if self._popup is not None and self._popup.isVisible():
            self._popup.update_value_and_meta(self._last_real, self._invalid)

    def enterEvent(self, e):
        super().enterEvent(e)
        if self._popup is None:
            self._popup = ParamHoverPopup(self)
            self._popup.set_param(self.spec)
        self._popup.update_value_and_meta(self._last_real, self._invalid)
        p = QCursor.pos()
        self._popup.move(p + QPoint(14, 14))
        self._popup.show()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        if self._popup is not None:
            self._popup.hide()


class ParamTextRow(QWidget):
    def __init__(self, spec: ParamSpec):
        super().__init__()
        self.spec = spec
        self._invalid = False
        self._value = spec.vmin
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(8)
        self.lbl_name = make_label(spec.label, bold=True, size=10, align=Qt.AlignLeft)
        self.lbl_val = make_label("-", bold=True, size=10, align=Qt.AlignRight)
        lay.addWidget(self.lbl_name, 1)
        lay.addWidget(self.lbl_val)

    def set_invalid(self, inv: bool):
        self._invalid = bool(inv)
        self._apply()

    def set_value(self, v: float):
        self._value = v
        self._apply()

    def _apply(self):
        s = self.spec
        if self._invalid:
            state, txt = "INVALID", "-"
        else:
            is_warn = (s.warning_hi is not None and self._value >= s.warning_hi) or (s.warning_lo is not None and self._value <= s.warning_lo)
            is_cau = (s.caution_hi is not None and self._value >= s.caution_hi) or (s.caution_lo is not None and self._value <= s.caution_lo)
            state = "WARNING" if is_warn else ("CAUTION" if is_cau else "NOMINAL")
            txt = f"{self._value:.{s.decimals}f} {s.unit}".strip()

        if state in ("WARNING", "INVALID"):
            bg, fg = "#FF3333", "#000000"
        elif state == "CAUTION":
            bg, fg = "#FFB000", "#000000"
        else:
            bg, fg = None, "#FFFFFF"

        if bg:
            self.setStyleSheet(f"background-color:{bg}; border-radius:4px;")
            name_fg = "#000000"
            val_fg = "#000000"
        else:
            self.setStyleSheet("background:transparent;")
            name_fg = "#FFFFFF"
            val_fg = "#FFFFFF"

        self.lbl_name.setStyleSheet(
            f"color:{name_fg}; font-size:10px; font-weight:bold; font-family:Consolas, monospace;"
        )
        self.lbl_val.setText(txt)
        self.lbl_val.setStyleSheet(
            f"color:{val_fg}; font-size:10px; font-weight:bold; font-family:Consolas, monospace;"
        )


class SpeedGauge(QWidget):
    def __init__(self, label, unit, vmin, vmax, caution_hi=None, warning_hi=None):
        super().__init__()
        self.label = label
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        self.caution_hi = caution_hi
        self.warning_hi = warning_hi
        self.value = vmin
        self.setMinimumSize(180, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v: float):
        self.value = clamp(v, self.vmin, self.vmax)
        self.update()

    def _color(self):
        v = self.value
        if self.warning_hi is not None and v >= self.warning_hi:
            return QColor("#FF3333")
        if self.caution_hi is not None and v >= self.caution_hi:
            return QColor("#FFB000")
        return QColor("#00FF66")

    def paintEvent(self, _):
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#000000"))

        margin = 16
        cx = w // 2
        cy = int(h * 0.78)
        r = min(w // 2 - margin, int(h * 0.62))

        start_deg = 225
        span_deg = 270
        rect = (cx - r, cy - r, 2 * r, 2 * r)

        pen = QPen(QColor("#888"))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawArc(*rect, start_deg * 16, -span_deg * 16)

        tick = QPen(QColor("#FFF"))
        tick.setWidth(2)
        p.setPen(tick)

        minor = 5
        total = 10 * minor
        for i in range(total + 1):
            t = i / total
            ang_deg = start_deg - (span_deg * t)
            ang = ang_deg * math.pi / 180
            is_major = (i % minor == 0)
            tl = 14 if is_major else 8
            cosv, sinv = math.cos(ang), math.sin(ang)
            xo = cx + (r - 2) * cosv
            yo = cy - (r - 2) * sinv
            xi = cx + (r - 2 - tl) * cosv
            yi = cy - (r - 2 - tl) * sinv
            p.drawLine(int(xi), int(yi), int(xo), int(yo))
            if is_major:
                val = self.vmin + (self.vmax - self.vmin) * t
                p.setFont(QFont("Consolas", 9, QFont.Bold))
                p.setPen(QPen(QColor("#CCC")))
                tx = cx + (r - 30) * cosv
                ty = cy - (r - 30) * sinv
                p.drawText(int(tx) - 14, int(ty) + 4, f"{int(round(val))}")
                p.setPen(tick)

        ratio = 0.0 if self.vmax == self.vmin else (self.value - self.vmin) / (self.vmax - self.vmin)
        ratio = clamp(ratio, 0.0, 1.0)
        ang_deg = start_deg - (span_deg * ratio)
        ang = ang_deg * math.pi / 180
        nl = r - 34
        nx = cx + nl * math.cos(ang)
        ny = cy - nl * math.sin(ang)

        # İbrenin kendisini yarı saydam (Alpha: 100) yapıyoruz
        needle_color = self._color()
        needle_color.setAlpha(100) 
        npen = QPen(needle_color)
        npen.setWidth(4)
        p.setPen(npen)
        p.drawLine(cx, cy, int(nx), int(ny))

        p.setBrush(QColor("#111"))
        p.setPen(QPen(QColor("#666")))
        p.drawEllipse(QPointF(cx, cy), 8, 8)

        p.setPen(QPen(QColor("#FFF")))
        p.setFont(QFont("Consolas", 11, QFont.Bold))
        p.drawText(0, 18, w, 18, Qt.AlignCenter, self.label)

        # Yazılar orijinal (doğru) konumlarına geri alındı
        p.setFont(QFont("Consolas", 18, QFont.Bold))
        p.setPen(QPen(self._color()))
        p.drawText(0, int(h * 0.40), w, 28, Qt.AlignCenter, f"{int(round(self.value))}")

        p.setFont(QFont("Consolas", 10))
        p.setPen(QPen(QColor("#CCC")))
        p.drawText(0, int(h * 0.52), w, 18, Qt.AlignCenter, self.unit)