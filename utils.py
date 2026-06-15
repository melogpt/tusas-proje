from typing import Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QLabel

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