"""
TUSAS TestLab — Parameter Configuration
========================================
Her parametrenin:
  - Kendi min/max range'i
  - Renk eşikleri (threshold → color mapping)
  - WCA'ya düşüp düşmeyeceği
  - Visual check listesi
  - Hata kategorisi

Bu config test motorunun merkezi kaynağıdır.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# ─── HATA KATEGORİLERİ ────────────────────────────────────────────────────────

class ErrorCategory(str, Enum):
    WCA_CRITICAL        = "WCA_CRITICAL"       # Kritik alarm — WCA'ya düşmeli
    COLOR_THRESHOLD     = "COLOR_THRESHOLD"    # Renk eşiği yanlış (60'da kırmızı yerine yeşil)
    VISUAL_SCALE        = "VISUAL_SCALE"       # Bar doluluk oranı numeric değerle uyuşmuyor
    UNIT_MISMATCH       = "UNIT_MISMATCH"      # Yanlış birim (°C yerine °F)
    STATE_MISMATCH      = "STATE_MISMATCH"     # Enum state yanlış (ANTI-ICE: AUTO beklendi, OFF geldi)
    BACKGROUND_COLOR    = "BACKGROUND_COLOR"   # Arka fon rengi güncellenmedi
    MISSING_SECTION     = "MISSING_SECTION"    # Panel/bölüm render edilmedi (boş alan)
    UNEXPECTED_TEXT     = "UNEXPECTED_TEXT"    # Tanımsız yazı/artefact (örn. "Sinem")
    LABEL_ERROR         = "LABEL_ERROR"        # Yanlış label metni
    WCA_SPURIOUS        = "WCA_SPURIOUS"       # WCA gereksiz yere çıkmış
    WCA_MISSING         = "WCA_MISSING"        # WCA çıkması gerekiyor ama yok
    WCA_WRONG_TEXT      = "WCA_WRONG_TEXT"     # WCA metni yanlış
    WCA_WRONG_COLOR     = "WCA_WRONG_COLOR"    # WCA rengi yanlış (kırmızı/sarı)
    WCA_DUPLICATE       = "WCA_DUPLICATE"      # WCA duplicate çıkıyor


class WcaFlag(str, Enum):
    REQUIRED     = "REQUIRED"      # Bu hata WCA'da görünmeli
    NOT_EXPECTED = "NOT_EXPECTED"  # Bu hata WCA'da görünmemeli
    OPTIONAL     = "OPTIONAL"      # WCA olabilir de olmayabilir de


# ─── RENK ARALIK TANIMI ───────────────────────────────────────────────────────

@dataclass
class ColorRange:
    min_val: float
    max_val: float
    color_name: str        # "green" | "orange" | "red" | "blue" | "yellow"
    color_hex: str         # Dominant hex renk (CV analizi için)
    wca_text: Optional[str] = None    # Bu range'de WCA'da çıkması beklenen metin
    wca_severity: Optional[str] = None  # "WARNING" | "CAUTION" | None


# ─── PARAMETRE KONFIG ─────────────────────────────────────────────────────────

@dataclass
class ParamConfig:
    key: str
    label: str
    unit: str
    param_type: str         # "numeric_bar" | "numeric_text" | "enum" | "gauge"
    vmin: float
    vmax: float
    visual_checks: List[str] = field(default_factory=list)
    # ["numeric", "bar_fill", "bar_color", "unit", "label", "background", "text"]
    color_ranges: List[ColorRange] = field(default_factory=list)
    wca_enabled: bool = True
    decimals: int = 1               # Gösterimde kullanılan ondalık basamak sayısı
    allowed_values: List[str] = field(default_factory=list)   # enum tipi için
    is_status_label: bool = False   # STATUS panelindeki label'lar

    def expected_color(self, value: float) -> Optional[str]:
        """Verilen değer için beklenen renk adını döndür."""
        for r in self.color_ranges:
            if r.min_val <= value < r.max_val:
                return r.color_name
        if self.color_ranges:
            last = self.color_ranges[-1]
            if value >= last.max_val:
                return last.color_name
        return None

    def expected_wca(self, value: float) -> Optional[str]:
        """Verilen değer için beklenen WCA metnini döndür."""
        for r in self.color_ranges:
            if r.min_val <= value < r.max_val and r.wca_text:
                return r.wca_text
        return None

    def expected_fill_ratio(self, value: float) -> float:
        """Verilen değer için beklenen bar doluluk oranını döndür (0.0-1.0)."""
        if self.vmax == self.vmin:
            return 0.0
        ratio = (value - self.vmin) / (self.vmax - self.vmin)
        return max(0.0, min(1.0, ratio))


# ─── TÜM PARAMETRELER ─────────────────────────────────────────────────────────

PARAM_CONFIGS: Dict[str, ParamConfig] = {

    # ══ ENGINE 1 ══════════════════════════════════════════════════════════════
    "E1_TRQ": ParamConfig(
        key="E1_TRQ", label="TRQ1", unit="%",
        param_type="numeric_bar", vmin=0, vmax=110,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 100, "green",  "#00FF66"),
            ColorRange(100, 105, "orange", "#FFB000", "TRQ1 HIGH", "CAUTION"),
            ColorRange(105, 110, "red",   "#FF3333", "TRQ1 HIGH", "WARNING"),
        ]
    ),
    "E1_TIT": ParamConfig(
        key="E1_TIT", label="TIT1", unit="°C",
        param_type="numeric_bar", vmin=0, vmax=950,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True, decimals=0,
        color_ranges=[
            ColorRange(0, 880, "green",  "#00FF66"),
            ColorRange(880, 920, "orange", "#FFB000", "TIT1 HIGH", "CAUTION"),
            ColorRange(920, 950, "red",   "#FF3333", "TIT1 HIGH", "WARNING"),
        ]
    ),
    "E1_NG": ParamConfig(
        key="E1_NG", label="NG1", unit="%",
        param_type="numeric_bar", vmin=0, vmax=110,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 102, "green",  "#00FF66"),
            ColorRange(102, 106, "orange", "#FFB000", "NG1 HIGH", "CAUTION"),
            ColorRange(106, 110, "red",   "#FF3333", "NG1 HIGH", "WARNING"),
        ]
    ),
    "E1_NP": ParamConfig(
        key="E1_NP", label="NP1", unit="%",
        param_type="numeric_bar", vmin=0, vmax=110,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 92,  "red",    "#FF3333", "NP1 LOW", "WARNING"),
            ColorRange(92, 96, "orange", "#FFB000", "NP1 LOW", "CAUTION"),
            ColorRange(96, 104, "green",  "#00FF66"),
            ColorRange(104, 107, "orange", "#FFB000", "NP1 HIGH", "CAUTION"),
            ColorRange(107, 110, "red",   "#FF3333", "NP1 HIGH", "WARNING"),
        ]
    ),
    "E1_OILP": ParamConfig(
        key="E1_OILP", label="OIL P1", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=120,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  25,  "red",    "#FF3333", "OIL P1 LOW", "WARNING"),
            ColorRange(25, 35,  "orange", "#FFB000", "OIL P1 LOW", "CAUTION"),
            ColorRange(35, 95,  "green",  "#00FF66"),
            ColorRange(95, 105, "orange", "#FFB000", "OIL P1 HIGH", "CAUTION"),
            ColorRange(105, 120, "red",   "#FF3333", "OIL P1 HIGH", "WARNING"),
        ]
    ),
    "E1_OILT": ParamConfig(
        key="E1_OILT", label="OIL T1", unit="°C",
        param_type="numeric_bar", vmin=0, vmax=160,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 120, "green",  "#00FF66"),
            ColorRange(120, 135, "orange", "#FFB000"),
            ColorRange(135, 160, "red",   "#FF3333"),
        ]
    ),

    # ══ ENGINE 2 ══════════════════════════════════════════════════════════════
    "E2_TRQ": ParamConfig(
        key="E2_TRQ", label="TRQ2", unit="%",
        param_type="numeric_bar", vmin=0, vmax=110,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 100, "green",  "#00FF66"),
            ColorRange(100, 105, "orange", "#FFB000", "TRQ2 HIGH", "CAUTION"),
            ColorRange(105, 110, "red",   "#FF3333", "TRQ2 HIGH", "WARNING"),
        ]
    ),
    "E2_TIT": ParamConfig(
        key="E2_TIT", label="TIT2", unit="°C",
        param_type="numeric_bar", vmin=0, vmax=950,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0, 880, "green",  "#00FF66"),
            ColorRange(880, 920, "orange", "#FFB000", "TIT2 HIGH", "CAUTION"),
            ColorRange(920, 950, "red",   "#FF3333", "TIT2 HIGH", "WARNING"),
        ]
    ),
    "E2_OILP": ParamConfig(
        key="E2_OILP", label="OIL P2", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=120,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  25,  "red",    "#FF3333", "OIL P2 LOW", "WARNING"),
            ColorRange(25, 35,  "orange", "#FFB000", "OIL P2 LOW", "CAUTION"),
            ColorRange(35, 95,  "green",  "#00FF66"),
            ColorRange(95, 105, "orange", "#FFB000"),
            ColorRange(105, 120, "red",   "#FF3333"),
        ]
    ),

    # ══ FUEL ══════════════════════════════════════════════════════════════════
    "FUEL_L": ParamConfig(
        key="FUEL_L", label="L TANK", unit="LBS",
        param_type="numeric_bar", vmin=0, vmax=600,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  40, "red",    "#FF3333", "L TANK LOW", "WARNING"),
            ColorRange(40, 80, "orange", "#FFB000", "L TANK LOW", "CAUTION"),
            ColorRange(80, 600, "green", "#00FF66"),
        ]
    ),
    "FUEL_R": ParamConfig(
        key="FUEL_R", label="R TANK", unit="LBS",
        param_type="numeric_bar", vmin=0, vmax=600,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  40, "red",    "#FF3333", "R TANK LOW", "WARNING"),
            ColorRange(40, 80, "orange", "#FFB000"),
            ColorRange(80, 600, "green", "#00FF66"),
        ]
    ),
    "FUEL_TOT": ParamConfig(
        key="FUEL_TOT", label="TOTAL", unit="LBS",
        param_type="numeric_bar", vmin=0, vmax=1200,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  100, "red",    "#FF3333", "FUEL LOW", "WARNING"),
            ColorRange(100, 180, "orange", "#FFB000"),
            ColorRange(180, 1200, "green", "#00FF66"),
        ]
    ),
    "FUEL_BAL": ParamConfig(
        key="FUEL_BAL", label="IMBAL", unit="LBS",
        param_type="numeric_bar", vmin=0, vmax=200,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  90,  "green",  "#00FF66"),
            ColorRange(90, 140, "orange", "#FFB000", "IMBAL HIGH", "CAUTION"),
            ColorRange(140, 200, "red",   "#FF3333", "IMBAL HIGH", "WARNING"),
        ]
    ),
    "FUEL_P": ParamConfig(
        key="FUEL_P", label="FUEL P", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=60,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  10, "red",    "#FF3333", "FUEL P LOW", "WARNING"),
            ColorRange(10, 18, "orange", "#FFB000", "FUEL P LOW", "CAUTION"),
            ColorRange(18, 60, "green",  "#00FF66"),
        ]
    ),

    # ══ ELECTRICAL ════════════════════════════════════════════════════════════
    "ELEC_GEN1V": ParamConfig(
        key="ELEC_GEN1V", label="GEN1 V", unit="V",
        param_type="numeric_text", vmin=0, vmax=32,
        visual_checks=["numeric", "unit", "background"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  22, "red",    "#FF3333", "GEN1 V LOW", "WARNING"),
            ColorRange(22, 25, "orange", "#FFB000", "GEN1 V LOW", "CAUTION"),
            ColorRange(25, 30, "green",  "#00FF66"),
            ColorRange(30, 31, "orange", "#FFB000"),
            ColorRange(31, 32, "red",   "#FF3333"),
        ]
    ),
    "ELEC_GEN2V": ParamConfig(
        key="ELEC_GEN2V", label="GEN2 V", unit="V",
        param_type="numeric_text", vmin=0, vmax=32,
        visual_checks=["numeric", "unit", "background"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  22, "red",    "#FF3333", "GEN2 V LOW", "WARNING"),
            ColorRange(22, 25, "orange", "#FFB000"),
            ColorRange(25, 30, "green",  "#00FF66"),
            ColorRange(30, 31, "orange", "#FFB000"),
            ColorRange(31, 32, "red",   "#FF3333"),
        ]
    ),
    "ELEC_ACF": ParamConfig(
        key="ELEC_ACF", label="AC FREQ", unit="Hz",
        param_type="numeric_text", vmin=0, vmax=500,
        visual_checks=["numeric", "unit", "background"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,   360, "red",    "#FF3333", "AC FREQ LOW", "WARNING"),
            ColorRange(360, 380, "orange", "#FFB000", "AC FREQ LOW", "CAUTION"),
            ColorRange(380, 420, "green",  "#00FF66"),
            ColorRange(420, 440, "orange", "#FFB000"),
            ColorRange(440, 500, "red",   "#FF3333"),
        ]
    ),
    "ELEC_BATTV": ParamConfig(
        key="ELEC_BATTV", label="BATT V", unit="V",
        param_type="numeric_text", vmin=0, vmax=32,
        visual_checks=["numeric", "unit", "background"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  22, "red",    "#FF3333"),
            ColorRange(22, 24, "orange", "#FFB000"),
            ColorRange(24, 32, "green",  "#00FF66"),
        ]
    ),
    "ELEC_DCBUS": ParamConfig(
        key="ELEC_DCBUS", label="DC BUS", unit="V",
        param_type="numeric_text", vmin=0, vmax=32,
        visual_checks=["numeric", "unit", "background"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  22, "red",    "#FF3333"),
            ColorRange(22, 25, "orange", "#FFB000"),
            ColorRange(25, 32, "green",  "#00FF66"),
        ]
    ),

    # ══ ENVIRONMENTAL ══════════════════════════════════════════════════════════
    "ENV_CABALT": ParamConfig(
        key="ENV_CABALT", label="CAB ALT", unit="FT",
        param_type="numeric_bar", vmin=0, vmax=12000,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,     8000,  "green",  "#00FF66"),
            ColorRange(8000,  10000, "orange", "#FFB000", "CAB ALT HIGH", "CAUTION"),
            ColorRange(10000, 12000, "red",    "#FF3333", "CAB ALT HIGH", "WARNING"),
        ]
    ),
    "ENV_SMK": ParamConfig(
        key="ENV_SMK", label="SMOKE", unit="%",
        param_type="numeric_bar", vmin=0, vmax=100,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  30, "green",  "#00FF66"),
            ColorRange(30, 50, "orange", "#FFB000", "SMOKE HIGH", "CAUTION"),
            ColorRange(50, 100, "red",   "#FF3333", "SMOKE HIGH", "WARNING"),
        ]
    ),
    "ENV_BLEED": ParamConfig(
        key="ENV_BLEED", label="BLEED", unit="%",
        param_type="numeric_bar", vmin=0, vmax=100,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  90, "green",  "#00FF66"),
            ColorRange(90, 97, "orange", "#FFB000", "BLEED HIGH", "CAUTION"),
            ColorRange(97, 100, "red",   "#FF3333", "BLEED HIGH", "WARNING"),
        ]
    ),
    "ENV_CABT": ParamConfig(
        key="ENV_CABT", label="CABIN T", unit="°C",
        param_type="numeric_bar", vmin=-20, vmax=60,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(-20, -5, "red",   "#FF3333"),
            ColorRange(-5,   0, "orange", "#FFB000"),
            ColorRange(0,   40, "green",  "#00FF66"),
            ColorRange(40,  48, "orange", "#FFB000"),
            ColorRange(48,  60, "red",   "#FF3333"),
        ]
    ),

    # ══ HYDRAULIC ══════════════════════════════════════════════════════════════
    "HYD_A_P": ParamConfig(
        key="HYD_A_P", label="SYS A P", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=3500,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,    1800, "red",    "#FF3333", "SYS A P LOW", "WARNING"),
            ColorRange(1800, 2200, "orange", "#FFB000", "SYS A P LOW", "CAUTION"),
            ColorRange(2200, 3500, "green",  "#00FF66"),
        ]
    ),
    "HYD_B_P": ParamConfig(
        key="HYD_B_P", label="SYS B P", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=3500,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,    1800, "red",    "#FF3333"),
            ColorRange(1800, 2200, "orange", "#FFB000"),
            ColorRange(2200, 3500, "green",  "#00FF66"),
        ]
    ),

    # ══ ROTOR / DRIVE ══════════════════════════════════════════════════════════
    "VIB_MR": ParamConfig(
        key="VIB_MR", label="MR VIB", unit="IPS",
        param_type="numeric_bar", vmin=0, vmax=10,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,   6.0, "green",  "#00FF66"),
            ColorRange(6.0, 7.5, "orange", "#FFB000", "MR VIB HIGH", "CAUTION"),
            ColorRange(7.5, 10,  "red",    "#FF3333", "MR VIB HIGH", "WARNING"),
        ]
    ),
    "VIB_TR": ParamConfig(
        key="VIB_TR", label="TR VIB", unit="IPS",
        param_type="numeric_bar", vmin=0, vmax=10,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,   6.0, "green",  "#00FF66"),
            ColorRange(6.0, 7.5, "orange", "#FFB000"),
            ColorRange(7.5, 10,  "red",    "#FF3333"),
        ]
    ),
    "RTR_NR": ParamConfig(
        key="RTR_NR", label="NR", unit="%",
        param_type="numeric_bar", vmin=0, vmax=110,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,   92,  "red",    "#FF3333", "NR LOW", "WARNING"),
            ColorRange(92,  96,  "orange", "#FFB000", "NR LOW", "CAUTION"),
            ColorRange(96,  104, "green",  "#00FF66"),
            ColorRange(104, 107, "orange", "#FFB000"),
            ColorRange(107, 110, "red",    "#FF3333"),
        ]
    ),
    "GBX_OILP": ParamConfig(
        key="GBX_OILP", label="GBX OIL P", unit="PSI",
        param_type="numeric_bar", vmin=0, vmax=200,
        visual_checks=["numeric", "bar_fill", "bar_color", "unit"],
        wca_enabled=True,
        color_ranges=[
            ColorRange(0,  35, "red",    "#FF3333"),
            ColorRange(35, 50, "orange", "#FFB000"),
            ColorRange(50, 200, "green", "#00FF66"),
        ]
    ),

    # ══ STATUS LABELS (enum) ══════════════════════════════════════════════════
    "ANTI_ICE": ParamConfig(
        key="ANTI_ICE", label="ANTI-ICE", unit="",
        param_type="enum", vmin=0, vmax=0,
        visual_checks=["text", "background"],
        wca_enabled=False,
        allowed_values=["OFF", "AUTO", "ON"],
        is_status_label=True,
    ),
    "PITOT_HEAT": ParamConfig(
        key="PITOT_HEAT", label="PITOT HEAT", unit="",
        param_type="enum", vmin=0, vmax=0,
        visual_checks=["text"],
        wca_enabled=False,
        allowed_values=["ON", "OFF"],
        is_status_label=True,
    ),
    "AUTOPILOT": ParamConfig(
        key="AUTOPILOT", label="AUTOPILOT", unit="",
        param_type="enum", vmin=0, vmax=0,
        visual_checks=["text"],
        wca_enabled=False,
        allowed_values=["ON", "OFF"],
        is_status_label=True,
    ),
}


# ─── WCA PANELİ İZİN VERİLEN METİNLER ────────────────────────────────────────
# Bu listede olmayan bir metin WCA'da görünürse → UNEXPECTED_TEXT hatası

WCA_ALLOWED_TEXTS: set = {
    "MONITORING ACTIVE", "FLT TIME", "DATA LINK: OK",
    "DEMO WARNING", "DEMO CAUTION",
    "TRQ1", "TRQ2", "TIT1", "TIT2", "NG1", "NG2", "NP1", "NP2",
    "OIL P1", "OIL P2", "OIL T1", "OIL T2", "P3 1", "P3 2",
    "L TANK", "R TANK", "TOTAL", "IMBAL", "FUEL P",
    "GEN1 V", "GEN2 V", "GEN1 A", "GEN2 A",
    "BATT V", "BATT A", "AC FREQ", "DC BUS",
    "CAB ALT", "CABIN T", "SMOKE", "BLEED", "HUM",
    "SYS A P", "SYS B P", "SYS A T", "SYS B T",
    "MR VIB", "TR VIB", "NR", "NG", "GBX OIL P",
    "SERVO", "TRIM",
    "INVALID", "HIGH", "LOW", "WARNING", "CAUTION", "ADVISORY",
    "IAS", "ALT", "V/S", "HDG",
}

# STATUS panelindeki label'lar için izin verilen değerler
STATUS_ALLOWED_VALUES: Dict[str, List[str]] = {
    "ANTI-ICE":      ["OFF", "AUTO", "ON"],
    "PITOT HEAT":    ["ON", "OFF"],
    "AUTOPILOT":     ["ON", "OFF"],
    "FD / NAV":      ["ARMED", "OFF", "NAV", "ON"],
    "MASTER CAUTION": ["ON", "OFF"],
    "WOW":           ["AIR", "GROUND"],
}
