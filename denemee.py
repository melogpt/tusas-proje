# Revised Flight Display (PyQt5)
# Implements: WCA interval text (no raw numbers), explicit severity text, label background by state,
# Engine1/Engine2 compartments with white divider, Electrical as text-only (no bars),
# INVALID overlay (red X) with '-' value, WCA fault timing gates + cap ~5 faults.

import sys, random, math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from PyQt5.QtCore import Qt, QTimer, QTime, QPoint, QPointF, QRect
from PyQt5.QtGui import QPainter, QPen, QFont, QColor, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QSizePolicy,
    QDialog, QPushButton, QScrollArea, QToolTip,
    QFrame
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_panel(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet("""
        QGroupBox { border: 1px solid #888; margin-top: 10px; color: white;
                    font-family: Consolas, monospace; font-size: 11px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
    """)
    return box


def make_label(text: str, align=Qt.AlignLeft, color="#FFFFFF", bold=False, size=11,
               bg: Optional[str] = None, pad="2px 6px", radius=3) -> QLabel:
    lbl = QLabel(text)
    weight = "bold" if bold else "normal"
    bg_css = f"background-color: {bg}; border-radius: {radius}px; padding: {pad};" if bg else ""
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size}px; font-weight: {weight}; "
        f"font-family: Consolas, monospace; {bg_css}"
    )
    lbl.setAlignment(align)
    return lbl


@dataclass
class ParamSpec:
    key: str
    label: str
    unit: str
    vmin: float
    vmax: float
    caution_hi: Optional[float] = None
    warning_hi: Optional[float] = None
    caution_lo: Optional[float] = None
    warning_lo: Optional[float] = None
    orientation: str = "V"
    decimals: int = 1
    min_warning_s: int = 5
    min_caution_s: int = 4


class BarVisual(QWidget):
    def __init__(self, spec: ParamSpec, big: bool = False):
        super().__init__()
        self.spec = spec
        self.big = big
        self._value = spec.vmin
        self._color = QColor("#00FF66")
        self._invalid = False

        if spec.orientation.upper() == "V":
            self.setFixedWidth(34 if big else 28)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setMinimumHeight(120 if big else 92)
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

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor("#050505"))

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
        root.addWidget(self.lbl_name)

        self.bar = BarVisual(spec, big=big)
        root.addWidget(self.bar, alignment=Qt.AlignHCenter)

        self.lbl_val = make_label("-", align=Qt.AlignCenter, size=11 if big else 10)
        root.addWidget(self.lbl_val)

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

    def set_invalid(self, inv: bool):
        self._invalid = bool(inv)
        self.bar.set_invalid(self._invalid)
        if self._invalid:
            self.lbl_val.setText("-")
            self._apply_label_bg("INVALID")
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
            self._update_popup()
            return

        self.lbl_val.setText(f"{real:.{s.decimals}f} {s.unit}".strip())
        self.bar.set_value(real)
        self.bar.set_color(self._color_for_value(real))
        self._apply_label_bg(self.get_state(real, False))
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

        bg_css = f"background-color:{bg}; border-radius:3px; padding:2px 6px;" if bg else "padding:0px;"
        self.lbl_name.setStyleSheet(f"color:{fg if bg else '#FFFFFF'}; font-size:10px; font-weight:bold; font-family:Consolas, monospace; {bg_css}")
        self.lbl_val.setText(txt)
        self.lbl_val.setStyleSheet(f"color:{('#000000' if bg else '#FFFFFF')}; font-size:10px; font-weight:bold; font-family:Consolas, monospace;")


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
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

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

        npen = QPen(self._color())
        npen.setWidth(4)
        p.setPen(npen)
        p.drawLine(cx, cy, int(nx), int(ny))

        p.setBrush(QColor("#111"))
        p.setPen(QPen(QColor("#666")))
        p.drawEllipse(QPointF(cx, cy), 8, 8)

        p.setPen(QPen(QColor("#FFF")))
        p.setFont(QFont("Consolas", 11, QFont.Bold))
        p.drawText(0, 18, w, 18, Qt.AlignCenter, self.label)

        p.setFont(QFont("Consolas", 18, QFont.Bold))
        p.setPen(QPen(self._color()))
        p.drawText(0, int(h * 0.40), w, 28, Qt.AlignCenter, f"{int(round(self.value))}")

        p.setFont(QFont("Consolas", 10))
        p.setPen(QPen(QColor("#CCC")))
        p.drawText(0, int(h * 0.52), w, 18, Qt.AlignCenter, self.unit)


@dataclass
class WcaEntry:
    key: str
    text: str
    severity: str
    first_seen_ms: int
    last_seen_ms: int
    count: int = 1


class WcaStore:
    def __init__(self):
        self.entries: List[WcaEntry] = []
        self._by: Dict[str, WcaEntry] = {}

    @staticmethod
    def rank(sev: str) -> int:
        return {"WARNING": 3, "CAUTION": 2, "ADVISORY": 1}.get(sev, 0)

    def upsert(self, now_ms: int, sev: str, key: str, text: str):
        if key in self._by:
            e = self._by[key]
            if self.rank(sev) > self.rank(e.severity):
                e.severity = sev
            e.text = text
            e.last_seen_ms = now_ms
            e.count += 1
        else:
            e = WcaEntry(key, text, sev, now_ms, now_ms, 1)
            self.entries.append(e)
            self._by[key] = e

    def snapshot_sorted(self) -> List[WcaEntry]:
        return sorted(self.entries, key=lambda e: (self.rank(e.severity), e.last_seen_ms, e.first_seen_ms), reverse=True)

    def counts(self) -> Tuple[int, int, int]:
        w = sum(1 for e in self.entries if e.severity == "WARNING")
        c = sum(1 for e in self.entries if e.severity == "CAUTION")
        a = sum(1 for e in self.entries if e.severity == "ADVISORY")
        return w, c, a


@dataclass
class FaultState:
    active: bool = False
    start_ms: int = 0
    last_ms: int = 0


class FaultGate:
    def __init__(self):
        self._s: Dict[str, FaultState] = {}

    def update(self, key: str, now_ms: int, active: bool) -> FaultState:
        st = self._s.get(key)
        if st is None:
            st = FaultState()
            self._s[key] = st
        if not active:
            st.active = False
            st.start_ms = 0
            st.last_ms = now_ms
            return st
        if not st.active:
            st.active = True
            st.start_ms = now_ms
            st.last_ms = now_ms
            return st
        st.last_ms = now_ms
        return st


class WcaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WCA - FULL LIST")
        self.setMinimumSize(760, 560)
        self.setStyleSheet("background-color:black;")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(make_label("WCA FULL LOG", align=Qt.AlignCenter, size=14, bold=True))

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #666; }
            QScrollBar:vertical { background:#111; width:14px; }
            QScrollBar::handle:vertical { background:#444; min-height:25px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
        """)
        root.addWidget(self.scroll, 1)

        self.body = QWidget()
        self.bl = QVBoxLayout(self.body)
        self.bl.setContentsMargins(10, 10, 10, 10)
        self.bl.setSpacing(8)
        self.scroll.setWidget(self.body)

        row = QHBoxLayout()
        root.addLayout(row)
        btn = QPushButton("CLOSE")
        btn.clicked.connect(self.close)
        btn.setStyleSheet("QPushButton{background:#222;color:white;border:1px solid #666;padding:8px 14px;font-family:Consolas, monospace;font-size:12px;font-weight:bold;} QPushButton:hover{border:1px solid #AAA;}")
        row.addStretch(1)
        row.addWidget(btn)

        self._dyn: List[QWidget] = []

    def set_entries(self, entries: List[WcaEntry], now_ms: int):
        for w in self._dyn:
            w.setParent(None)
            w.deleteLater()
        self._dyn.clear()

        for e in entries:
            age_s = max(0, (now_ms - e.last_seen_ms) // 1000)
            chip = "#00FF66" if e.severity == "ADVISORY" else ("#FFB000" if e.severity == "CAUTION" else "#FF3333")

            row = QFrame()
            row.setStyleSheet("QFrame{border:1px solid #444;background:#050505;}")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(10)

            tag = QFrame()
            tag.setFixedWidth(10)
            tag.setStyleSheet(f"QFrame{{background:{chip};border:1px solid #222;}}")
            lay.addWidget(tag)

            txt = make_label(f"{e.severity}: {e.text}", color="#FFFFFF", size=11, bold=(e.severity == "WARNING"))
            txt.setWordWrap(True)
            lay.addWidget(txt, 1)

            meta = make_label(f"x{e.count}  age:{age_s}s", color="#AAAAAA", size=10)
            meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(meta)

            self.bl.addWidget(row)
            self._dyn.append(row)


class FlightDisplay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flight Systems Display")
        self.setMinimumSize(1600, 860)

        self.param_widgets: Dict[str, ParamBar] = {}
        self.text_widgets: Dict[str, ParamTextRow] = {}
        self.vals: Dict[str, float] = {}
        self.invalid: Dict[str, bool] = {}

        self.wca = WcaStore()
        self.wca_dialog = WcaDialog(self)
        self.fgate = FaultGate()

        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color:black;")
        main = QVBoxLayout(central)

        tframe = QFrame()
        tframe.setStyleSheet("QFrame{border:1px solid #888;}")
        tf = QVBoxLayout(tframe)
        tf.addWidget(make_label("FLIGHT SYSTEMS DISPLAY", align=Qt.AlignCenter, size=20, bold=True))
        main.addWidget(tframe)

        top = QHBoxLayout()
        main.addLayout(top)
        top.addWidget(self._status_panel())
        top.addWidget(self._systems_panel())
        top.addWidget(self._time_panel())

        mid = QHBoxLayout()
        main.addLayout(mid, 1)
        mid.addWidget(self._engine_cluster(), 3)
        right = QVBoxLayout()
        mid.addLayout(right, 2)
        right.addWidget(self._flight_panel())
        right.addWidget(self._env_panel())
        right.addWidget(self._nav_panel())

        bottom = QHBoxLayout()
        main.addLayout(bottom)
        bottom.addWidget(self._fuel_panel())
        bottom.addWidget(self._electrical_panel())
        bottom.addWidget(self._hyd_panel())
        bottom.addWidget(self._drive_panel())
        bottom.addWidget(self._wca_panel())

        QToolTip.setFont(QFont("Consolas", 10))
        self._start()

    def _status_panel(self):
        box = make_panel("STATUS")
        g = QGridLayout(box)
        g.addWidget(make_label("WOW"), 0, 0)
        self.lbl_wow = make_label("AIR", color="#FFFF00", bold=True)
        g.addWidget(self.lbl_wow, 0, 1, alignment=Qt.AlignRight)
        g.addWidget(make_label("PITOT HEAT"), 1, 0)
        self.lbl_pitot = make_label("ON", color="#00FF66", bold=True)
        g.addWidget(self.lbl_pitot, 1, 1, alignment=Qt.AlignRight)
        g.addWidget(make_label("ANTI-ICE"), 2, 0)
        self.lbl_anti = make_label("AUTO", color="#00FF66", bold=True)
        g.addWidget(self.lbl_anti, 2, 1, alignment=Qt.AlignRight)
        return box

    def _systems_panel(self):
        box = make_panel("SYSTEMS STATUS")
        g = QGridLayout(box)
        g.addWidget(make_label("AUTOPILOT"), 0, 0)
        self.lbl_ap = make_label("ON", color="#00FF66", bold=True)
        g.addWidget(self.lbl_ap, 0, 1, alignment=Qt.AlignRight)
        g.addWidget(make_label("FD / NAV"), 1, 0)
        self.lbl_fd = make_label("ARMED", color="#00FF66", bold=True)
        g.addWidget(self.lbl_fd, 1, 1, alignment=Qt.AlignRight)
        g.addWidget(make_label("MASTER CAUTION"), 2, 0)
        self.lbl_mc = make_label("OFF", color="#00FF66", bold=True)
        g.addWidget(self.lbl_mc, 2, 1, alignment=Qt.AlignRight)
        return box

    def _time_panel(self):
        box = make_panel("FLIGHT TIME")
        v = QVBoxLayout(box)
        self.lbl_time = make_label("00:00:00", align=Qt.AlignCenter, size=18, bold=True)
        v.addWidget(self.lbl_time)
        return box

    def _add_grid(self, grid: QGridLayout, specs: List[ParamSpec], cols: int, big: bool):
        r = c = 0
        for i, s in enumerate(specs):
            w = ParamBar(s, big=big)
            self.param_widgets[s.key] = w
            self.vals[s.key] = s.vmin
            self.invalid[s.key] = False
            grid.addWidget(w, r, c)
            c += 1
            if (i + 1) % cols == 0:
                r += 1
                c = 0

    def _add_text(self, lay: QVBoxLayout, specs: List[ParamSpec]):
        for s in specs:
            w = ParamTextRow(s)
            self.text_widgets[s.key] = w
            self.vals[s.key] = s.vmin
            self.invalid[s.key] = False
            lay.addWidget(w)

    def _engine_cluster(self):
        box = make_panel("PROPULSION / ENGINES")
        outer = QVBoxLayout(box)
        outer.addWidget(make_label("ENGINE & POWERPLANT PARAMETERS", align=Qt.AlignCenter, size=14, bold=True, color="#FFFF33"))

        row = QHBoxLayout()
        outer.addLayout(row)
        left = QVBoxLayout()
        right = QVBoxLayout()
        row.addLayout(left, 1)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setLineWidth(1)
        divider.setStyleSheet("QFrame{background:#FFFFFF;}")
        row.addWidget(divider)

        row.addLayout(right, 1)

        e1 = make_panel("ENGINE 1")
        e2 = make_panel("ENGINE 2")
        left.addWidget(e1)
        right.addWidget(e2)

        g1 = QGridLayout(e1)
        g2 = QGridLayout(e2)

        self._add_grid(g1, [
            ParamSpec("E1_NG", "NG1", "%", 0, 110, caution_hi=102, warning_hi=106, decimals=1),
            ParamSpec("E1_TIT", "TIT1", "°C", 0, 950, caution_hi=880, warning_hi=920, decimals=0),
            ParamSpec("E1_TRQ", "TRQ1", "%", 0, 110, caution_hi=100, warning_hi=105, decimals=1),
            ParamSpec("E1_NP", "NP1", "%", 0, 110, caution_lo=96, warning_lo=92, caution_hi=104, warning_hi=107, decimals=1),
        ], cols=4, big=True)

        self._add_grid(g2, [
            ParamSpec("E2_NG", "NG2", "%", 0, 110, caution_hi=102, warning_hi=106, decimals=1),
            ParamSpec("E2_TIT", "TIT2", "°C", 0, 950, caution_hi=880, warning_hi=920, decimals=0),
            ParamSpec("E2_TRQ", "TRQ2", "%", 0, 110, caution_hi=100, warning_hi=105, decimals=1),
            ParamSpec("E2_NP", "NP2", "%", 0, 110, caution_lo=96, warning_lo=92, caution_hi=104, warning_hi=107, decimals=1),
        ], cols=4, big=True)

        return box

    def _flight_panel(self):
        box = make_panel("FLIGHT / AVIONICS")
        outer = QHBoxLayout(box)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        self.gauge_ias = SpeedGauge("IAS", "KT", 0, 220, caution_hi=200, warning_hi=210)
        outer.addWidget(self.gauge_ias)

        holder = QWidget()
        g = QGridLayout(holder)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)

        self._add_grid(g, [
            ParamSpec("FLT_IAS", "IAS", "KT", 0, 220, caution_hi=200, warning_hi=210, decimals=0),
            ParamSpec("FLT_ALT", "ALT", "FT", 0, 20000, caution_hi=18000, warning_hi=19000, decimals=0),
            ParamSpec("FLT_VS", "V/S", "FPM", -2000, 2000, caution_hi=1500, warning_hi=1800, caution_lo=-1500, warning_lo=-1800, decimals=0),
            ParamSpec("FLT_HDG", "HDG", "°", 0, 360, decimals=0),
            ParamSpec("FLT_GS", "G/S", "KT", 0, 220, decimals=0),
            ParamSpec("NAV_GPS", "GPS ACC", "m", 0, 20, caution_hi=12, warning_hi=16, decimals=1),
            ParamSpec("NAV_DME", "DME", "NM", 0, 200, decimals=1),
            ParamSpec("AP_TRIM", "TRIM", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
        ], cols=4, big=False)

        outer.addWidget(holder, 1)
        return box

    def _env_panel(self):
        box = make_panel("ENVIRONMENTAL / BLEED")
        g = QGridLayout(box)
        self._add_grid(g, [
            ParamSpec("ENV_CABT", "CABIN T", "°C", -20, 60, caution_hi=40, warning_hi=48, caution_lo=0, warning_lo=-5, decimals=0),
            ParamSpec("ENV_OAT", "OAT", "°C", -50, 50, decimals=0),
            ParamSpec("ENV_CABALT", "CAB ALT", "FT", 0, 12000, caution_hi=8000, warning_hi=10000, decimals=0),
            ParamSpec("ENV_DIFF", "DIFF P", "PSI", 0, 10, caution_hi=8.5, warning_hi=9.2, decimals=1),
            ParamSpec("ENV_BLEED", "BLEED", "%", 0, 100, caution_hi=90, warning_hi=97, decimals=0),
            ParamSpec("ENV_DEFOG", "DEFOG", "%", 0, 100, decimals=0),
            ParamSpec("ENV_HUM", "HUM", "%", 0, 100, caution_hi=85, warning_hi=92, decimals=0),
            ParamSpec("ENV_SMK", "SMOKE", "%", 0, 100, caution_hi=30, warning_hi=50, decimals=0),
        ], cols=4, big=False)
        return box

    def _nav_panel(self):
        box = make_panel("NAV / RADIOS")
        g = QGridLayout(box)
        self._add_grid(g, [
            ParamSpec("NAV1_FREQ", "NAV1", "MHz", 108.0, 118.0, decimals=2),
            ParamSpec("NAV2_FREQ", "NAV2", "MHz", 108.0, 118.0, decimals=2),
            ParamSpec("COM1_FREQ", "COM1", "MHz", 118.0, 137.0, decimals=2),
            ParamSpec("COM2_FREQ", "COM2", "MHz", 118.0, 137.0, decimals=2),
            ParamSpec("ADF_BRG", "ADF BRG", "°", 0, 360, decimals=0),
            ParamSpec("VOR_DEV", "VOR DEV", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
            ParamSpec("LOC_DEV", "LOC DEV", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
            ParamSpec("GS_DEV", "GS DEV", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
        ], cols=4, big=False)
        return box

    def _fuel_panel(self):
        box = make_panel("FUEL")
        g = QGridLayout(box)
        self._add_grid(g, [
            ParamSpec("FUEL_L", "L TANK", "LBS", 0, 600, caution_lo=80, warning_lo=40, decimals=0),
            ParamSpec("FUEL_R", "R TANK", "LBS", 0, 600, caution_lo=80, warning_lo=40, decimals=0),
            ParamSpec("FUEL_TOT", "TOTAL", "LBS", 0, 1200, caution_lo=180, warning_lo=100, decimals=0),
            ParamSpec("FUEL_BAL", "IMBAL", "LBS", 0, 200, caution_hi=90, warning_hi=140, decimals=0),
            ParamSpec("FUEL_P", "FUEL P", "PSI", 0, 60, caution_lo=18, warning_lo=10, decimals=0),
            ParamSpec("FUEL_TEMP", "FUEL T", "°C", -40, 50, caution_lo=-30, warning_lo=-35, decimals=0),
        ], cols=3, big=False)
        return box

    def _electrical_panel(self):
        box = make_panel("ELECTRICAL (AC/DC)")
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        self._add_text(v, [
            ParamSpec("ELEC_GEN1V", "GEN1 V", "V", 0, 32, caution_lo=25, warning_lo=22, caution_hi=30, warning_hi=31, decimals=1),
            ParamSpec("ELEC_GEN2V", "GEN2 V", "V", 0, 32, caution_lo=25, warning_lo=22, caution_hi=30, warning_hi=31, decimals=1),
            ParamSpec("ELEC_GEN1A", "GEN1 A", "A", 0, 400, caution_hi=320, warning_hi=360, decimals=0),
            ParamSpec("ELEC_GEN2A", "GEN2 A", "A", 0, 400, caution_hi=320, warning_hi=360, decimals=0),
            ParamSpec("ELEC_BATTV", "BATT V", "V", 0, 32, caution_lo=24, warning_lo=22, decimals=1),
            ParamSpec("ELEC_BATTA", "BATT A", "A", -200, 200, caution_hi=140, warning_hi=170, caution_lo=-140, warning_lo=-170, decimals=0),
            ParamSpec("ELEC_ACF", "AC FREQ", "Hz", 0, 500, caution_lo=380, warning_lo=360, caution_hi=420, warning_hi=440, decimals=0),
            ParamSpec("ELEC_DCBUS", "DC BUS", "V", 0, 32, caution_lo=25, warning_lo=22, decimals=1),
        ])
        return box

    def _hyd_panel(self):
        box = make_panel("HYDRAULICS / FLT CTRL")
        g = QGridLayout(box)
        self._add_grid(g, [
            ParamSpec("HYD_A_P", "SYS A P", "PSI", 0, 3500, caution_lo=2200, warning_lo=1800, decimals=0),
            ParamSpec("HYD_B_P", "SYS B P", "PSI", 0, 3500, caution_lo=2200, warning_lo=1800, decimals=0),
            ParamSpec("HYD_A_T", "SYS A T", "°C", 0, 140, caution_hi=110, warning_hi=125, decimals=0),
            ParamSpec("HYD_B_T", "SYS B T", "°C", 0, 140, caution_hi=110, warning_hi=125, decimals=0),
            ParamSpec("CTL_SERVO", "SERVO", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
            ParamSpec("CTL_TRIM", "TRIM", "%", 0, 100, caution_hi=85, warning_hi=95, decimals=0),
        ], cols=3, big=False)
        return box

    def _drive_panel(self):
        box = make_panel("DRIVE / ROTOR")
        g = QGridLayout(box)
        self._add_grid(g, [
            ParamSpec("RTR_NR", "NR", "%", 0, 110, caution_lo=96, warning_lo=92, caution_hi=104, warning_hi=107, decimals=1),
            ParamSpec("RTR_NG", "NG", "%", 0, 110, caution_hi=104, warning_hi=107, decimals=1),
            ParamSpec("GBX_OILP", "GBX OIL P", "PSI", 0, 200, caution_lo=50, warning_lo=35, decimals=0),
            ParamSpec("GBX_OILT", "GBX OIL T", "°C", 0, 160, caution_hi=120, warning_hi=135, decimals=0),
            ParamSpec("VIB_MR", "MR VIB", "IPS", 0, 10, caution_hi=6.0, warning_hi=7.5, decimals=1),
        ], cols=3, big=False)
        return box

    def _wca_panel(self):
        box = make_panel("WCA")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        top = QHBoxLayout()
        v.addLayout(top)
        self.lbl_counts = make_label("W:0  C:0  A:0", bold=True, size=11)
        top.addWidget(self.lbl_counts)
        top.addStretch(1)

        btn = QPushButton("MORE")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("QPushButton{background:#222;color:white;border:1px solid #666;padding:6px 12px;font-family:Consolas, monospace;font-size:11px;font-weight:bold;} QPushButton:hover{border:1px solid #AAA;}")
        btn.clicked.connect(self._open_wca)
        top.addWidget(btn)

        self.wca_frame = QFrame()
        self.wca_frame.setStyleSheet("QFrame{background:#00FF66;border:1px solid #333;}")
        v.addWidget(self.wca_frame, 1)

        wl = QVBoxLayout(self.wca_frame)
        wl.setContentsMargins(8, 8, 8, 8)
        wl.setSpacing(6)

        self.wca_title = make_label("SYSTEM STATUS", align=Qt.AlignCenter, color="black", bold=True, size=12)
        wl.addWidget(self.wca_title)

        self.wca_scroll = QScrollArea()
        self.wca_scroll.setWidgetResizable(True)
        self.wca_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.wca_scroll.setStyleSheet("QScrollArea{border:1px solid #222;background:transparent;} QScrollBar:vertical{background:#111;width:12px;} QScrollBar::handle:vertical{background:#444;min-height:25px;} QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0px;}")
        wl.addWidget(self.wca_scroll, 1)

        self.wca_list = QWidget()
        self.wca_lay = QVBoxLayout(self.wca_list)
        self.wca_lay.setContentsMargins(8, 8, 8, 8)
        self.wca_lay.setSpacing(6)
        self.wca_scroll.setWidget(self.wca_list)

        return box

    def _open_wca(self):
        now = self._now_ms()
        self.wca_dialog.set_entries(self.wca.snapshot_sorted(), now)
        self.wca_dialog.exec_()

    def _clear_vbox(self, lay: QVBoxLayout):
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _render_wca(self, now_ms: int):
        entries = self.wca.snapshot_sorted()
        w, c, a = self.wca.counts()
        self.lbl_counts.setText(f"W:{w}  C:{c}  A:{a}")

        top = entries[0].severity if entries else "ADVISORY"
        if top == "WARNING":
            self.wca_frame.setStyleSheet("QFrame{background:#FF3333;border:1px solid #333;}")
            self.wca_title.setText("WARNING LIST")
        elif top == "CAUTION":
            self.wca_frame.setStyleSheet("QFrame{background:#FFB000;border:1px solid #333;}")
            self.wca_title.setText("CAUTION LIST")
        else:
            self.wca_frame.setStyleSheet("QFrame{background:#00FF66;border:1px solid #333;}")
            self.wca_title.setText("SYSTEM STATUS")

        self.wca_title.setStyleSheet("color:black;font-weight:bold;font-family:Consolas, monospace;")

        self._clear_vbox(self.wca_lay)
        if not entries:
            entries = [WcaEntry("ADV_MON", "MONITORING ACTIVE", "ADVISORY", now_ms, now_ms, 1)]
        entries = entries[:5]

        for e in entries:
            bg = "#FF3333" if e.severity == "WARNING" else ("#FFB000" if e.severity == "CAUTION" else "#00FF66")
            lbl = make_label(f"{e.severity}: {e.text}", color="#000000", bg=bg, size=11, bold=(e.severity == "WARNING"), pad="3px 6px", radius=4)
            lbl.setWordWrap(True)
            self.wca_lay.addWidget(lbl)

    def _now_ms(self) -> int:
        return int(QTime(0, 0, 0).msecsTo(self.elapsed))

    def _start(self):
        self.elapsed = QTime(0, 0, 0)
        self.t_time = QTimer(self)
        self.t_time.timeout.connect(self._tick_time)
        self.t_time.start(1000)

        self._sim_init()

        self.t_sim = QTimer(self)
        self.t_sim.timeout.connect(self._tick_sim)
        self.t_sim.start(700)

    def _tick_time(self):
        self.elapsed = self.elapsed.addSecs(1)
        self.lbl_time.setText(self.elapsed.toString("hh:mm:ss"))

    def _apply_ui(self):
        for k, w in self.param_widgets.items():
            w.set_invalid(self.invalid.get(k, False))
            w.set_value(self.vals.get(k, w.spec.vmin))
        for k, w in self.text_widgets.items():
            w.set_invalid(self.invalid.get(k, False))
            w.set_value(self.vals.get(k, w.spec.vmin))
        self.gauge_ias.set_value(self.vals.get("FLT_IAS", 0))

    def _sim_init(self):
        init = {
            "E1_NG": 84, "E1_TIT": 650, "E1_TRQ": 78, "E1_NP": 100.0,
            "E2_NG": 83, "E2_TIT": 645, "E2_TRQ": 77, "E2_NP": 100.0,
            "FLT_IAS": 120, "FLT_ALT": 4500, "FLT_VS": 0, "FLT_HDG": 275, "FLT_GS": 115, "NAV_GPS": 2.4, "NAV_DME": 18.2, "AP_TRIM": 40,
            "ENV_CABT": 21, "ENV_OAT": 5, "ENV_CABALT": 1200, "ENV_DIFF": 1.2, "ENV_BLEED": 55, "ENV_DEFOG": 10, "ENV_HUM": 35, "ENV_SMK": 0,
            "FUEL_L": 380, "FUEL_R": 370, "FUEL_TOT": 750, "FUEL_BAL": 10, "FUEL_P": 38, "FUEL_TEMP": -5,
            "ELEC_GEN1V": 28.2, "ELEC_GEN2V": 28.1, "ELEC_GEN1A": 120, "ELEC_GEN2A": 130, "ELEC_BATTV": 27.6, "ELEC_BATTA": 15, "ELEC_ACF": 400, "ELEC_DCBUS": 28.0,
            "HYD_A_P": 3000, "HYD_B_P": 2950, "HYD_A_T": 75, "HYD_B_T": 78, "CTL_SERVO": 35, "CTL_TRIM": 42,
            "RTR_NR": 100.2, "RTR_NG": 99.8, "GBX_OILP": 110, "GBX_OILT": 92, "VIB_MR": 2.4,
            "NAV1_FREQ": 110.30, "NAV2_FREQ": 114.70, "COM1_FREQ": 124.85, "COM2_FREQ": 121.90,
            "ADF_BRG": 45, "VOR_DEV": 12, "LOC_DEV": 8, "GS_DEV": 10,
        }
        for k, v in init.items():
            self.vals[k] = float(v)

        self._apply_ui()

        now = self._now_ms()
        self.wca.upsert(now, "ADVISORY", "ADV_MON", "MONITORING ACTIVE")
        self.wca.upsert(now, "CAUTION", "CAU_DEMO", "DEMO CAUTION (MINIMIZED)")
        self.wca.upsert(now, "WARNING", "WRN_DEMO", "DEMO WARNING (MINIMIZED)")
        self._render_wca(now)

    def _nudge(self, k, step, lo=None, hi=None):
        v = self.vals.get(k, 0.0) + random.uniform(-step, step)
        if lo is not None and hi is not None:
            v = clamp(v, lo, hi)
        self.vals[k] = v

    def _wca_interval_text(self, spec: ParamSpec, value: float, invalid: bool) -> str:
        if invalid:
            return f"{spec.label} INVALID"
        if spec.warning_hi is not None and value >= spec.warning_hi:
            return f"{spec.label} HIGH >{spec.warning_hi:.{spec.decimals}f}{spec.unit}".strip()
        if spec.warning_lo is not None and value <= spec.warning_lo:
            return f"{spec.label} LOW <{spec.warning_lo:.{spec.decimals}f}{spec.unit}".strip()
        if spec.caution_hi is not None and value >= spec.caution_hi:
            return f"{spec.label} HIGH >{spec.caution_hi:.{spec.decimals}f}{spec.unit}".strip()
        if spec.caution_lo is not None and value <= spec.caution_lo:
            return f"{spec.label} LOW <{spec.caution_lo:.{spec.decimals}f}{spec.unit}".strip()
        return f"{spec.label} OK"

    def _tick_sim(self):
        # invalid injection
        for k in list(self.invalid.keys()):
            if random.random() < 0.003:
                self.invalid[k] = True
            elif self.invalid[k] and random.random() < 0.25:
                self.invalid[k] = False

        # dynamics
        self._nudge("FLT_IAS", 2.5, 0, 220)
        self._nudge("FLT_GS", 2.0, 0, 220)
        self._nudge("FLT_VS", 120, -2000, 2000)
        self._nudge("FLT_ALT", 40, 0, 20000)
        self._nudge("FLT_HDG", 2.0, 0, 360)
        self._nudge("NAV_GPS", 0.3, 0, 20)
        self._nudge("NAV_DME", 0.2, 0, 200)
        self._nudge("AP_TRIM", 1.8, 0, 100)

        self._nudge("E1_TRQ", 1.2, 0, 110)
        self._nudge("E2_TRQ", 1.2, 0, 110)
        self.vals["E1_NG"] = clamp(60 + self.vals["E1_TRQ"] * 0.35 + random.uniform(-1.5, 1.5), 0, 110)
        self.vals["E2_NG"] = clamp(60 + self.vals["E2_TRQ"] * 0.35 + random.uniform(-1.5, 1.5), 0, 110)
        self.vals["E1_TIT"] = clamp(480 + self.vals["E1_TRQ"] * 3.0 + random.uniform(-8, 8), 0, 950)
        self.vals["E2_TIT"] = clamp(480 + self.vals["E2_TRQ"] * 3.0 + random.uniform(-8, 8), 0, 950)
        self._nudge("E1_NP", 0.6, 0, 110)
        self._nudge("E2_NP", 0.6, 0, 110)

        self.vals["ENV_OAT"] = clamp(self.vals["ENV_OAT"] + random.uniform(-0.2, 0.2), -50, 50)
        self.vals["ENV_CABT"] = clamp(self.vals["ENV_CABT"] + random.uniform(-0.3, 0.4), -20, 60)
        self.vals["ENV_CABALT"] = clamp(1000 + (self.vals["FLT_ALT"] * 0.08) + random.uniform(-80, 80), 0, 12000)
        self.vals["ENV_DIFF"] = clamp(0.8 + (self.vals["FLT_ALT"] / 20000) * 3.0 + random.uniform(-0.1, 0.1), 0, 10)
        self.vals["ENV_BLEED"] = clamp(self.vals["ENV_BLEED"] + random.uniform(-2, 3), 0, 100)
        self.vals["ENV_DEFOG"] = clamp(self.vals["ENV_DEFOG"] + random.uniform(-2, 2), 0, 100)
        self.vals["ENV_HUM"] = clamp(self.vals["ENV_HUM"] + random.uniform(-2, 2), 0, 100)

        if random.random() < 0.02:
            self.vals["ENV_SMK"] = clamp(self.vals["ENV_SMK"] + random.uniform(10, 25), 0, 100)
        else:
            self.vals["ENV_SMK"] = clamp(self.vals["ENV_SMK"] - random.uniform(2, 5), 0, 100)

        self._nudge("FUEL_P", 2.0, 0, 60)
        self._nudge("FUEL_TEMP", 0.3, -40, 50)
        self.vals["FUEL_L"] = clamp(self.vals["FUEL_L"] - random.uniform(0.05, 0.2), 0, 600)
        self.vals["FUEL_R"] = clamp(self.vals["FUEL_R"] - random.uniform(0.05, 0.2), 0, 600)
        self.vals["FUEL_TOT"] = clamp(self.vals["FUEL_L"] + self.vals["FUEL_R"], 0, 1200)
        self.vals["FUEL_BAL"] = abs(self.vals["FUEL_L"] - self.vals["FUEL_R"])

        self.vals["ELEC_GEN1V"] = clamp(28.0 + random.uniform(-0.4, 0.4), 0, 32)
        self.vals["ELEC_GEN2V"] = clamp(28.0 + random.uniform(-0.4, 0.4), 0, 32)
        self.vals["ELEC_GEN1A"] = clamp(90 + random.uniform(-15, 25) + (self.vals["ENV_BLEED"] * 0.8), 0, 400)
        self.vals["ELEC_GEN2A"] = clamp(100 + random.uniform(-15, 25) + (self.vals["ENV_BLEED"] * 0.8), 0, 400)
        self.vals["ELEC_BATTV"] = clamp(27.5 + random.uniform(-0.3, 0.3), 0, 32)
        self.vals["ELEC_BATTA"] = clamp(10 + random.uniform(-20, 20), -200, 200)
        self.vals["ELEC_ACF"] = clamp(400 + random.uniform(-8, 8), 0, 500)
        self.vals["ELEC_DCBUS"] = clamp(28.0 + random.uniform(-0.5, 0.5), 0, 32)

        self.vals["RTR_NR"] = clamp(100 + random.uniform(-0.6, 0.6), 0, 110)
        self.vals["RTR_NG"] = clamp(100 + random.uniform(-0.6, 0.6), 0, 110)
        self.vals["GBX_OILP"] = clamp(105 + random.uniform(-8, 8), 0, 200)
        self.vals["GBX_OILT"] = clamp(85 + random.uniform(-2, 2), 0, 160)
        self.vals["VIB_MR"] = clamp(2.0 + random.uniform(-0.5, 0.7), 0, 10)

        self._nudge("ADF_BRG", 3.0, 0, 360)
        self._nudge("VOR_DEV", 3.0, 0, 100)
        self._nudge("LOC_DEV", 3.0, 0, 100)
        self._nudge("GS_DEV", 3.0, 0, 100)

        self.vals["NAV1_FREQ"] = clamp(self.vals["NAV1_FREQ"] + random.uniform(-0.01, 0.01), 108.0, 118.0)
        self.vals["NAV2_FREQ"] = clamp(self.vals["NAV2_FREQ"] + random.uniform(-0.01, 0.01), 108.0, 118.0)
        self.vals["COM1_FREQ"] = clamp(self.vals["COM1_FREQ"] + random.uniform(-0.01, 0.01), 118.0, 137.0)
        self.vals["COM2_FREQ"] = clamp(self.vals["COM2_FREQ"] + random.uniform(-0.01, 0.01), 118.0, 137.0)

        self.lbl_wow.setText("AIR" if random.random() > 0.05 else "GROUND")
        self.lbl_ap.setText("ON" if random.random() > 0.08 else "OFF")

        now = self._now_ms()

        # stable advisories
        self.wca.upsert(now, "ADVISORY", "ADV_MON", "MONITORING ACTIVE")
        self.wca.upsert(now, "ADVISORY", "ADV_TIME", f"FLT TIME {self.elapsed.toString('hh:mm:ss')}")
        self.wca.upsert(now, "ADVISORY", "ADV_LINK", "DATA LINK: OK")

        candidates: List[Tuple[int, str, str, str]] = []

        def push(rank, sev, key, text):
            candidates.append((rank, sev, key, text))

        # bars
        for key, w in self.param_widgets.items():
            spec = w.spec
            val = self.vals.get(key, spec.vmin)
            inv = self.invalid.get(key, False)
            state = w.get_state(val, inv)

            if state == "INVALID":
                st = self.fgate.update(f"INV_{key}", now, True)
                if (now - st.start_ms) >= spec.min_warning_s * 1000:
                    push(3, "WARNING", f"INV_{key}", self._wca_interval_text(spec, val, True))
                continue
            else:
                self.fgate.update(f"INV_{key}", now, False)

            if state == "WARNING":
                st = self.fgate.update(f"WRN_{key}", now, True)
                if (now - st.start_ms) >= spec.min_warning_s * 1000:
                    push(3, "WARNING", f"WRN_{key}", self._wca_interval_text(spec, val, False))
            else:
                self.fgate.update(f"WRN_{key}", now, False)

            if state == "CAUTION":
                st = self.fgate.update(f"CAU_{key}", now, True)
                if (now - st.start_ms) >= spec.min_caution_s * 1000:
                    push(2, "CAUTION", f"CAU_{key}", self._wca_interval_text(spec, val, False))
            else:
                self.fgate.update(f"CAU_{key}", now, False)

        # electrical text
        for key, w in self.text_widgets.items():
            spec = w.spec
            val = self.vals.get(key, spec.vmin)
            inv = self.invalid.get(key, False)

            if inv:
                st = self.fgate.update(f"INV_{key}", now, True)
                if (now - st.start_ms) >= spec.min_warning_s * 1000:
                    push(3, "WARNING", f"INV_{key}", f"{spec.label} INVALID")
                continue
            else:
                self.fgate.update(f"INV_{key}", now, False)

            is_warn = (spec.warning_hi is not None and val >= spec.warning_hi) or (spec.warning_lo is not None and val <= spec.warning_lo)
            is_cau = (spec.caution_hi is not None and val >= spec.caution_hi) or (spec.caution_lo is not None and val <= spec.caution_lo)

            if is_warn:
                st = self.fgate.update(f"WRN_{key}", now, True)
                if (now - st.start_ms) >= spec.min_warning_s * 1000:
                    push(3, "WARNING", f"WRN_{key}", self._wca_interval_text(spec, val, False))
            else:
                self.fgate.update(f"WRN_{key}", now, False)

            if (not is_warn) and is_cau:
                st = self.fgate.update(f"CAU_{key}", now, True)
                if (now - st.start_ms) >= spec.min_caution_s * 1000:
                    push(2, "CAUTION", f"CAU_{key}", self._wca_interval_text(spec, val, False))
            else:
                self.fgate.update(f"CAU_{key}", now, False)

        # keep faults small
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:5]

        for _, sev, k, text in candidates:
            self.wca.upsert(now, sev, k, text)

        has_warn = any(sev == "WARNING" for _, sev, _, _ in candidates)
        has_cau = any(sev == "CAUTION" for _, sev, _, _ in candidates)

        if has_warn:
            self.lbl_mc.setText("ON")
            self.lbl_mc.setStyleSheet("color:#FF3333;font-family:Consolas, monospace;font-weight:bold;")
        elif has_cau:
            self.lbl_mc.setText("ON")
            self.lbl_mc.setStyleSheet("color:#FFB000;font-family:Consolas, monospace;font-weight:bold;")
        else:
            self.lbl_mc.setText("OFF")
            self.lbl_mc.setStyleSheet("color:#00FF66;font-family:Consolas, monospace;font-weight:bold;")

        self._apply_ui()
        self._render_wca(now)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FlightDisplay()
    win.show()
    sys.exit(app.exec_())