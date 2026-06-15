import sys
import random
from typing import Dict, List, Tuple

from PyQt5.QtCore import Qt, QTimer, QTime, QThread, QMetaObject, Q_ARG
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QPushButton, QScrollArea, QToolTip
)

from utils import clamp, make_panel, make_label
from models import ParamSpec, WcaStore, FaultGate, WcaEntry
from widgets import ParamBar, ParamTextRow, SpeedGauge
from dialogs import WcaDialog
try:
    from ses_asistani import DinleyiciThread, SesOynatici
except ImportError:
    DinleyiciThread, SesOynatici = None, None


class FlightDisplay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flight Systems Display")
        self.resize(1280, 720)
        
        self.ses_ikaz_aktif = True
        self._ikaz_sayaci = 0

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

        # YENİ LAYOUT YAPISI: Üst Gövde (Sol ve Sağ olarak ikiye ayrılıyor)
        upper_body = QHBoxLayout()
        main.addLayout(upper_body, 1)

        # SOL TARAF (Oran: 3) - Üst Paneller ve Motor Panelleri burada
        left_side = QVBoxLayout()
        upper_body.addLayout(left_side, 3)

        top_panels = QHBoxLayout()
        left_side.addLayout(top_panels, 0)
        top_panels.addWidget(self._status_panel())
        top_panels.addWidget(self._systems_panel())
        top_panels.addWidget(self._time_panel())

        left_side.addWidget(self._engine_cluster(), 1)

        # SAĞ TARAF (Oran: 2) - Uçuş, Çevre ve Navigasyon Panelleri tepeye kadar uzanır
        right_side = QVBoxLayout()
        upper_body.addLayout(right_side, 2)
        right_side.addWidget(self._flight_panel())
        right_side.addWidget(self._env_panel())
        right_side.addWidget(self._nav_panel())

        # ALT TARAF
        bottom = QHBoxLayout()
        main.addLayout(bottom)
        bottom.addWidget(self._fuel_panel(), 10)
        bottom.addWidget(self._electrical_panel(), 5)
        bottom.addWidget(self._hyd_panel(), 10)
        bottom.addWidget(self._drive_panel(), 10)
        bottom.addWidget(self._wca_panel(), 20)

        QToolTip.setFont(QFont("Consolas", 10))
        self._start()
        self._ses_sistemini_baslat()

    def _ses_sistemini_baslat(self):
        if DinleyiciThread is None or SesOynatici is None:
            return
        try:
            self.ses_oynatici = SesOynatici()
            self.ses_oynaticisi_thread = QThread()
            self.ses_oynatici.moveToThread(self.ses_oynaticisi_thread)
            self.ses_oynaticisi_thread.start()

            self.dinleyici = DinleyiciThread(self)
            self.dinleyici.ses_duyuldu.connect(self._komut_isleyicide)
            self.dinleyici.start()
        except Exception as e:
            print("Ses sistemi baslatilamadi:", e)

    def _komut_isleyicide(self, komut):
        if "status" in komut or "report" in komut or "durum ne" in komut or "rapor ver" in komut:
            w, c, a = self.wca.counts()
            
            if w > 0 or c > 0 or a > 0:
                cevap = "System Status: ATTENTION. "
                cevap += "The following alerts are active: "
                hatalar = []
                for entry in self.wca.snapshot_sorted():
                    hatalar.append(f"{entry.text} ({entry.severity})")
                
                if len(hatalar) > 3:
                    cevap += ", ".join(hatalar[:3]) + f", and {len(hatalar) - 3} more."
                else:
                    cevap += ", ".join(hatalar) + "."
                
                cevap += " All other systems are nominal."
            else:
                cevap = "System Status: PASS. All systems including engine indicators, autopilot, and electrical sensors have passed the test. We are in nominal flight condition."
            
            QMetaObject.invokeMethod(self.ses_oynatici, "konus", 
                                            Qt.QueuedConnection, 
                                            Q_ARG(str, cevap))

    def _status_panel(self):
        box = make_panel("STATUS")
        g = QGridLayout(box)
        g.addWidget(make_label("WOW"), 0, 0)
        self.lbl_wow = make_label("AIR", color="#00FF66", bold=True)
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
        self.lbl_mc = make_label("OFF", color="#555555", bold=True)
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
            ParamSpec("E1_OILP", "OIL P1", "PSI", 0, 120, caution_lo=35, warning_lo=25, caution_hi=95, warning_hi=105, decimals=0),
            ParamSpec("E1_OILT", "OIL T1", "°C", 0, 160, caution_hi=120, warning_hi=135, decimals=0),
            ParamSpec("E1_OILQ", "OIL Q1", "%", 0, 100, caution_lo=35, warning_lo=25, decimals=0),
            ParamSpec("E1_P3", "P3 1", "PSI", 0, 120, caution_lo=40, warning_lo=30, caution_hi=105, warning_hi=112, decimals=0),
        ], cols=4, big=True)

        self._add_grid(g2, [
            ParamSpec("E2_NG", "NG2", "%", 0, 110, caution_hi=102, warning_hi=106, decimals=1),
            ParamSpec("E2_TIT", "TIT2", "°C", 0, 950, caution_hi=880, warning_hi=920, decimals=0),
            ParamSpec("E2_TRQ", "TRQ2", "%", 0, 110, caution_hi=100, warning_hi=105, decimals=1),
            ParamSpec("E2_NP", "NP2", "%", 0, 110, caution_lo=96, warning_lo=92, caution_hi=104, warning_hi=107, decimals=1),
            ParamSpec("E2_OILP", "OIL P2", "PSI", 0, 120, caution_lo=35, warning_lo=25, caution_hi=95, warning_hi=105, decimals=0),
            ParamSpec("E2_OILT", "OIL T2", "°C", 0, 160, caution_hi=120, warning_hi=135, decimals=0),
            ParamSpec("E2_OILQ", "OIL Q2", "%", 0, 100, caution_lo=35, warning_lo=25, decimals=0),
            ParamSpec("E2_P3", "P3 2", "PSI", 0, 120, caution_lo=40, warning_lo=30, caution_hi=105, warning_hi=112, decimals=0),
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
            ParamSpec("VIB_TR", "TR VIB", "IPS", 0, 10, caution_hi=6.0, warning_hi=7.5, decimals=1),
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

        self.btn_mute = QPushButton("MUTE")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setCursor(Qt.PointingHandCursor)
        self.btn_mute.setStyleSheet(
            "QPushButton{background:#222;color:white;border:1px solid #666;padding:6px 12px;"
            "font-family:Consolas, monospace;font-size:11px;font-weight:bold;}"
            "QPushButton:checked{background:#AA0000; color:white; border:1px solid #FF0000;}"
            "QPushButton:hover{border:1px solid #AAA;}"
        )
        self.btn_mute.clicked.connect(self._toggle_mute)
        top.addWidget(self.btn_mute)

        btn = QPushButton("MORE")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton{background:#222;color:white;border:1px solid #666;padding:6px 12px;"
            "font-family:Consolas, monospace;font-size:11px;font-weight:bold;}"
            "QPushButton:hover{border:1px solid #AAA;}"
        )
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
        self.wca_scroll.setStyleSheet(
            "QScrollArea{border:1px solid #222;background:transparent;}"
            "QScrollBar:vertical{background:#111;width:12px;}"
            "QScrollBar::handle:vertical{background:#444;min-height:25px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0px;}"
        )
        wl.addWidget(self.wca_scroll, 1)

        self.wca_list = QWidget()
        self.wca_lay = QVBoxLayout(self.wca_list)
        self.wca_lay.setContentsMargins(8, 8, 8, 8)
        self.wca_lay.setSpacing(6)
        self.wca_scroll.setWidget(self.wca_list)

        return box

    def _toggle_mute(self, checked):
        self.ses_ikaz_aktif = not checked
        self.btn_mute.setText("MUTED" if checked else "MUTE")
        if checked:
            QMetaObject.invokeMethod(self.ses_oynatici, "sustur", Qt.QueuedConnection)

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

        picked: List[WcaEntry] = []
        used = set()
        by_sev = {"WARNING": [], "CAUTION": [], "ADVISORY": []}
        for e in entries:
            if e.severity in by_sev:
                by_sev[e.severity].append(e)
        for sev in ("WARNING", "CAUTION", "ADVISORY"):
            if by_sev[sev]:
                e = by_sev[sev][0]
                picked.append(e)
                used.add(e.key)
        for e in entries:
            if len(picked) >= 5:
                break
            if e.key in used:
                continue
            picked.append(e)
            used.add(e.key)
        entries = picked[:5]

        for e in entries:
            bg = "#FF3333" if e.severity == "WARNING" else ("#FFB000" if e.severity == "CAUTION" else "#00FF66")
            dur_s = max(0, (e.last_seen_ms - e.first_seen_ms) // 1000)
            lbl = make_label(
                f"{e.severity}: {e.text}  t+{dur_s}s",
                color="#000000",
                bg=bg,
                size=11,
                bold=(e.severity == "WARNING"),
                pad="3px 6px",
                radius=4,
            )
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

        if getattr(self, 'ses_ikaz_aktif', False) and getattr(self, 'ses_oynatici', None):
            self._ikaz_sayaci += 1
            if self._ikaz_sayaci >= 3:
                sorted_entries = self.wca.snapshot_sorted()
                top_warning = next((e for e in sorted_entries if e.severity == "WARNING"), None)
                
                if top_warning:
                    QMetaObject.invokeMethod(self.ses_oynatici, "konus", Qt.QueuedConnection, Q_ARG(str, f"Warning. {top_warning.text}"))
                self._ikaz_sayaci = 0

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
            "E1_OILP": 78, "E1_OILT": 92, "E1_OILQ": 78, "E1_P3": 82,
            "E2_OILP": 79, "E2_OILT": 93, "E2_OILQ": 79, "E2_P3": 83,

            "FLT_IAS": 120, "FLT_ALT": 4500, "FLT_VS": 0, "FLT_HDG": 275, "FLT_GS": 115, "NAV_GPS": 2.4, "NAV_DME": 18.2, "AP_TRIM": 40,
            "ENV_CABT": 21, "ENV_OAT": 5, "ENV_CABALT": 1200, "ENV_DIFF": 1.2, "ENV_BLEED": 55, "ENV_DEFOG": 10, "ENV_HUM": 35, "ENV_SMK": 0,
            "FUEL_L": 380, "FUEL_R": 370, "FUEL_TOT": 750, "FUEL_BAL": 10, "FUEL_P": 38, "FUEL_TEMP": -5,
            "ELEC_GEN1V": 28.2, "ELEC_GEN2V": 28.1, "ELEC_GEN1A": 120, "ELEC_GEN2A": 130, "ELEC_BATTV": 27.6, "ELEC_BATTA": 15, "ELEC_ACF": 400, "ELEC_DCBUS": 28.0,
            "HYD_A_P": 3000, "HYD_B_P": 2950, "HYD_A_T": 75, "HYD_B_T": 78, "CTL_SERVO": 35, "CTL_TRIM": 42,
            "RTR_NR": 100.2, "RTR_NG": 99.8, "GBX_OILP": 110, "GBX_OILT": 92, "VIB_MR": 2.4, "VIB_TR": 1.9,
            "NAV1_FREQ": 110.30, "NAV2_FREQ": 114.70, "COM1_FREQ": 124.85, "COM2_FREQ": 121.90,
            "ADF_BRG": 45, "VOR_DEV": 12, "LOC_DEV": 8, "GS_DEV": 10,
        }
        for k, v in init.items():
            self.vals[k] = float(v)

        self._apply_ui()

        now = self._now_ms()
        self.wca.upsert(now, "ADVISORY", "ADV_MON", "MONITORING ACTIVE")
        self.wca.upsert(now, "CAUTION", "CAU_DEMO", "DEMO CAUTION")
        self.wca.upsert(now, "WARNING", "WRN_DEMO", "DEMO WARNING")
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
        for k in list(self.invalid.keys()):
            if random.random() < 0.003:
                self.invalid[k] = True
            elif self.invalid[k] and random.random() < 0.25:
                self.invalid[k] = False

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

        self.vals["E1_OILP"] = clamp(80 + random.uniform(-6, 6) - (self.vals["E1_TIT"] - 600) * 0.02, 0, 120)
        self.vals["E2_OILP"] = clamp(80 + random.uniform(-6, 6) - (self.vals["E2_TIT"] - 600) * 0.02, 0, 120)
        self.vals["E1_OILT"] = clamp(85 + (self.vals["E1_TIT"] - 500) * 0.06 + random.uniform(-2, 2), 0, 160)
        self.vals["E2_OILT"] = clamp(85 + (self.vals["E2_TIT"] - 500) * 0.06 + random.uniform(-2, 2), 0, 160)
        self.vals["E1_OILQ"] = clamp(self.vals.get("E1_OILQ", 80) + random.uniform(-0.15, 0.05), 0, 100)
        self.vals["E2_OILQ"] = clamp(self.vals.get("E2_OILQ", 80) + random.uniform(-0.15, 0.05), 0, 100)
        self.vals["E1_P3"] = clamp(30 + (self.vals["E1_NG"] * 0.7) + random.uniform(-3, 3), 0, 120)
        self.vals["E2_P3"] = clamp(30 + (self.vals["E2_NG"] * 0.7) + random.uniform(-3, 3), 0, 120)

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
        self.vals["VIB_TR"] = clamp(1.8 + random.uniform(-0.5, 0.7), 0, 10)

        self._nudge("ADF_BRG", 3.0, 0, 360)
        self._nudge("VOR_DEV", 3.0, 0, 100)
        self._nudge("LOC_DEV", 3.0, 0, 100)
        self._nudge("GS_DEV", 3.0, 0, 100)

        self.vals["NAV1_FREQ"] = clamp(self.vals["NAV1_FREQ"] + random.uniform(-0.01, 0.01), 108.0, 118.0)
        self.vals["NAV2_FREQ"] = clamp(self.vals["NAV2_FREQ"] + random.uniform(-0.01, 0.01), 108.0, 118.0)
        self.vals["COM1_FREQ"] = clamp(self.vals["COM1_FREQ"] + random.uniform(-0.01, 0.01), 118.0, 137.0)
        self.vals["COM2_FREQ"] = clamp(self.vals["COM2_FREQ"] + random.uniform(-0.01, 0.01), 118.0, 137.0)

        is_air = random.random() > 0.05
        self.lbl_wow.setText("AIR" if is_air else "GROUND")
        self.lbl_wow.setStyleSheet(f"color:{'#00FF66' if is_air else '#FFFFFF'};font-weight:bold;")
        
        is_ap_on = random.random() > 0.08
        self.lbl_ap.setText("ON" if is_ap_on else "OFF")
        self.lbl_ap.setStyleSheet(f"color:{'#00FF66' if is_ap_on else '#555555'};font-weight:bold;")

        now = self._now_ms()

        self.wca.upsert(now, "ADVISORY", "ADV_MON", "MONITORING ACTIVE")
        self.wca.upsert(now, "ADVISORY", "ADV_TIME", f"FLT TIME {self.elapsed.toString('hh:mm:ss')}")
        self.wca.upsert(now, "ADVISORY", "ADV_LINK", "DATA LINK: OK")

        candidates: List[Tuple[int, str, str, str]] = []

        def push(rank, sev, key, text):
            candidates.append((rank, sev, key, text))

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
            self.lbl_mc.setStyleSheet("color:#555555;font-family:Consolas, monospace;font-weight:bold;")

        self._apply_ui()
        self._render_wca(now)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FlightDisplay()
    win.showMaximized()
    sys.exit(app.exec_())