from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QScrollArea, QPushButton, QFrame
)

from utils import make_label
from models import WcaEntry


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
        btn.setStyleSheet(
            "QPushButton{background:#222;color:white;border:1px solid #666;padding:8px 14px;"
            "font-family:Consolas, monospace;font-size:12px;font-weight:bold;}"
            "QPushButton:hover{border:1px solid #AAA;}"
        )
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
            dur_s = max(0, (e.last_seen_ms - e.first_seen_ms) // 1000)
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

            txt = make_label(
                f"{e.severity}: {e.text}",
                color="#FFFFFF",
                size=11,
                bold=(e.severity == "WARNING"),
            )
            txt.setWordWrap(True)
            lay.addWidget(txt, 1)

            meta = make_label(f"x{e.count}  age:{age_s}s  dur:{dur_s}s", color="#AAAAAA", size=10)
            meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(meta)

            self.bl.addWidget(row)
            self._dyn.append(row)