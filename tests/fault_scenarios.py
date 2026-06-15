"""
TUSAŞ TestLab — Fault Injection Engine
=======================================
Deterministik hata senaryoları tanımlar.
Her senaryo; hangi parametrelerin ne değere zorlanacağını,
beklenen WCA mesajlarını ve beklenen görsel durumları içerir.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FaultScenario:
    """Tek bir test senaryosu tanımı."""
    id: str
    name: str
    category: str                          # "ENGINE" | "FUEL" | "ELECTRICAL" | "ENVIRONMENTAL" | "HYDRAULIC" | "ROTOR"
    severity: str                          # "WARNING" | "CAUTION" | "ADVISORY"
    description: str
    inject: Dict[str, float]              # param_key -> zorlanacak değer
    invalid_params: List[str] = field(default_factory=list)   # INVALID işaretlenecek paramlar
    expected_wca_keys: List[str] = field(default_factory=list)  # WCA'da görünmesi beklenen key prefixleri
    expected_wca_texts: List[str] = field(default_factory=list) # WCA metninde geçmesi beklenen kelimeler
    expected_mc_state: str = "OFF"         # Master Caution beklenen durum: "OFF" | "CAUTION" | "WARNING"
    ai_vision_prompt: str = ""             # Claude Vision'a gönderilecek soru
    time_advance_secs: int = 6             # FaultGate için ileri sarılacak süre (sn)
    tick_count: int = 2                    # Kaç kez _tick_sim çağrılacak


# ─── SENARYO KATALOğU ─────────────────────────────────────────────────────────

SCENARIOS: List[FaultScenario] = [

    # ── ENGINE ──────────────────────────────────────────────────────────────────
    FaultScenario(
        id="ENG_001",
        name="Engine 1 Torque Overload",
        category="ENGINE",
        severity="WARNING",
        description="Motor 1 torku uyarı limitini (%105) aşıyor. WCA kırmızı, MC ON bekleniyor.",
        inject={"E1_TRQ": 108.0, "E2_TRQ": 78.0},
        expected_wca_texts=["TRQ1", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Bu uçuş ekranında: "
            "1) WCA (Warning/Caution/Advisory) paneli kırmızı mı? "
            "2) Engine 1 Torque barı kırmızı renkte mi? "
            "3) Master Caution 'ON' yazıyor mu? "
            "Her soru için EVET veya HAYIR ile yanıtla."
        ),
    ),

    FaultScenario(
        id="ENG_002",
        name="Engine 2 High Turbine Temperature",
        category="ENGINE",
        severity="WARNING",
        description="Motor 2 türbin sıcaklığı kritik sınırı (920°C) aşıyor.",
        inject={"E2_TIT": 935.0, "E1_TIT": 650.0},
        expected_wca_texts=["TIT2", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Bu uçuş ekranında Engine 2 TIT (Turbine Inlet Temperature) parametresi "
            "kırmızı uyarı rengiyle gösteriliyor mu? "
            "WCA listesinde kırmızı renkli bir uyarı var mı? EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="ENG_003",
        name="Engine 1 Oil Pressure Low",
        category="ENGINE",
        severity="WARNING",
        description="Motor 1 yağ basıncı kritik düşük (25 PSI altı).",
        inject={"E1_OILP": 18.0},
        expected_wca_texts=["OIL P1", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Ekranda Engine 1 Oil Pressure (OIL P1) değeri kırmızı/uyarı durumunda mı? "
            "EVET veya HAYIR."
        ),
    ),

    FaultScenario(
        id="ENG_004",
        name="Both Engine NP Caution",
        category="ENGINE",
        severity="CAUTION",
        description="Her iki motorda pervane devri (NP) düşük — sarı ihtiyat.",
        inject={"E1_NP": 91.0, "E2_NP": 91.0},
        expected_wca_texts=["NP"],
        expected_mc_state="CAUTION",
        ai_vision_prompt=(
            "WCA paneli sarı (CAUTION) renk tonunda mı görünüyor? "
            "Master Caution 'ON' yazıyor mu? EVET/HAYIR."
        ),
    ),

    # ── FUEL ────────────────────────────────────────────────────────────────────
    FaultScenario(
        id="FUEL_001",
        name="Critical Fuel Low — Both Tanks",
        category="FUEL",
        severity="WARNING",
        description="Her iki yakıt tankı kritik seviyede (40 LBS altı). Acil iniş senaryosu.",
        inject={"FUEL_L": 25.0, "FUEL_R": 28.0, "FUEL_TOT": 53.0},
        expected_wca_texts=["L TANK", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Yakıt panelinde (FUEL bölümü) L TANK ve R TANK değerleri kırmızı/uyarı "
            "rengiyle gösteriliyor mu? WCA listesinde FUEL ile ilgili uyarı var mı? "
            "EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="FUEL_002",
        name="Fuel Imbalance Warning",
        category="FUEL",
        severity="WARNING",
        description="Sağ-sol tank dengesizliği 140 LBS üstü — yüksek imbalans.",
        inject={"FUEL_L": 400.0, "FUEL_R": 150.0, "FUEL_BAL": 250.0},
        expected_wca_texts=["IMBAL", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Yakıt dengesizlik (IMBAL) göstergesi kırmızı renkte mi? EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="FUEL_003",
        name="Fuel Pressure Caution",
        category="FUEL",
        severity="CAUTION",
        description="Yakıt basıncı düşük ihtiyat bölgesi (10–18 PSI arası).",
        inject={"FUEL_P": 14.0},
        expected_wca_texts=["FUEL P", "LOW"],
        expected_mc_state="CAUTION",
        ai_vision_prompt=(
            "Yakıt basıncı (FUEL P) göstergesi sarı/ihtiyat rengiyle mi gösteriliyor? "
            "EVET/HAYIR."
        ),
    ),

    # ── ELECTRICAL ──────────────────────────────────────────────────────────────
    FaultScenario(
        id="ELEC_001",
        name="Generator 1 Voltage Critical Low",
        category="ELECTRICAL",
        severity="WARNING",
        description="Jeneratör 1 voltajı 22V altına düştü — güç arızası.",
        inject={"ELEC_GEN1V": 19.5},
        expected_wca_texts=["GEN1 V", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Elektrik panelinde GEN1 V değeri kırmızı uyarı rengiyle gösteriliyor mu? "
            "EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="ELEC_002",
        name="AC Frequency Out of Range",
        category="ELECTRICAL",
        severity="WARNING",
        description="AC frekansı kabul edilemez seviyede — 360 Hz altı.",
        inject={"ELEC_ACF": 345.0},
        expected_wca_texts=["AC FREQ", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "AC FREQ parametresi kritik değerde mi gösteriliyor? EVET/HAYIR."
        ),
    ),

    # ── ENVIRONMENTAL ───────────────────────────────────────────────────────────
    FaultScenario(
        id="ENV_001",
        name="Cabin Altitude Emergency",
        category="ENVIRONMENTAL",
        severity="WARNING",
        description="Kabin basınç yüksekliği 10.000 FT üstü — dekompresyon.",
        inject={"ENV_CABALT": 10500.0, "ENV_DIFF": 1.0},
        expected_wca_texts=["CAB ALT", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Çevre panelinde CAB ALT değeri kırmızı uyarı rengiyle gösteriliyor mu? "
            "EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="ENV_002",
        name="Smoke Detection Alert",
        category="ENVIRONMENTAL",
        severity="WARNING",
        description="Duman algılama sensörü yüksek değer — kabin yangın uyarısı.",
        inject={"ENV_SMK": 65.0},
        expected_wca_texts=["SMOKE", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "SMOKE (Duman) göstergesi kırmızı/uyarı rengiyle gösteriliyor mu? "
            "WCA'da smoke ile ilgili uyarı var mı? EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="ENV_003",
        name="Bleed Air Overload",
        category="ENVIRONMENTAL",
        severity="WARNING",
        description="Bleed air sistemi kapasitesi aşıldı (%97 üstü).",
        inject={"ENV_BLEED": 105.0},
        expected_wca_texts=["BLEED", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "BLEED göstergesi kırmızı/uyarı rengiyle gösteriliyor mu? EVET/HAYIR."
        ),
    ),

    # ── HYDRAULIC ───────────────────────────────────────────────────────────────
    FaultScenario(
        id="HYD_001",
        name="Hydraulic System A Pressure Loss",
        category="HYDRAULIC",
        severity="WARNING",
        description="Hidrolik Sistem A basıncı kritik düşük (1800 PSI altı).",
        inject={"HYD_A_P": 1500.0},
        expected_wca_texts=["SYS A P", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Hidrolik panelinde SYS A P değeri kırmızı/uyarı rengiyle gösteriliyor mu? "
            "EVET/HAYIR."
        ),
    ),

    # ── ROTOR / DRIVE ───────────────────────────────────────────────────────────
    FaultScenario(
        id="RTR_001",
        name="Main Rotor Vibration High",
        category="ROTOR",
        severity="WARNING",
        description="Ana rotor vibrasyon seviyesi kritik (7.5 IPS üstü).",
        inject={"VIB_MR": 8.2},
        expected_wca_texts=["MR VIB", "HIGH"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "DRIVE/ROTOR panelinde MR VIB değeri kırmızı uyarı rengiyle gösteriliyor mu? "
            "EVET/HAYIR."
        ),
    ),

    FaultScenario(
        id="RTR_002",
        name="Rotor RPM Underspeed",
        category="ROTOR",
        severity="WARNING",
        description="Rotor devri kritik düşük (%92 altı) — uçuş emniyeti riski.",
        inject={"RTR_NR": 88.0},
        expected_wca_texts=["NR", "LOW"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "NR (Rotor RPM) göstergesi kırmızı/uyarı rengiyle mi gösteriliyor? "
            "EVET/HAYIR."
        ),
    ),

    # ── MULTI-SYSTEM CASCADING ───────────────────────────────────────────────────
    FaultScenario(
        id="CASCADE_001",
        name="Cascading Engine + Fuel Failure",
        category="ENGINE",
        severity="WARNING",
        description="Eş zamanlı Motor 1 + Yakıt kritik arızası. En kötü durum testi.",
        inject={
            "E1_TRQ": 108.0,
            "E1_OILP": 15.0,
            "FUEL_L": 20.0,
            "FUEL_R": 18.0,
            "FUEL_TOT": 38.0,
        },
        expected_wca_texts=["TRQ1", "OIL P1", "L TANK"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Bu kritik senaryoda: "
            "1) WCA paneli tamamen kırmızı mı? "
            "2) Birden fazla sistem için uyarı gösteriliyor mu? "
            "3) Master Caution 'ON' ve kırmızı mı? "
            "Her soru için EVET/HAYIR ile yanıtla."
        ),
        time_advance_secs=8,
        tick_count=3,
    ),

    FaultScenario(
        id="CASCADE_002",
        name="Full Electrical Blackout Simulation",
        category="ELECTRICAL",
        severity="WARNING",
        description="Tüm elektrik üretim sistemleri arızalı simülasyonu.",
        inject={
            "ELEC_GEN1V": 18.0,
            "ELEC_GEN2V": 17.5,
            "ELEC_BATTV": 20.0,
            "ELEC_DCBUS": 17.0,
        },
        expected_wca_texts=["GEN1 V", "GEN2 V"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Elektrik panelinde birden fazla voltaj değeri kırmızı/kritik gösteriliyor mu? "
            "WCA listesinde elektrik arızası uyarısı var mı? EVET/HAYIR."
        ),
        time_advance_secs=8,
        tick_count=3,
    ),

    # ── SENSOR INVALID ───────────────────────────────────────────────────────────
    FaultScenario(
        id="SENS_001",
        name="Multiple Sensor Invalid State",
        category="ENGINE",
        severity="WARNING",
        description="Birden fazla sensör INVALID durumuna geçiyor — veri bütünlüğü testi.",
        inject={"E1_NG": 84.0},
        invalid_params=["E1_NG", "E1_TIT", "E2_NG"],
        expected_wca_texts=["INVALID"],
        expected_mc_state="WARNING",
        ai_vision_prompt=(
            "Engine panelinde çarpı işareti (X) veya INVALID göstergesi olan "
            "parametreler var mı? WCA'da INVALID uyarısı görünüyor mu? EVET/HAYIR."
        ),
    ),

    # ── NOMINAL (baseline) ──────────────────────────────────────────────────────
    FaultScenario(
        id="NOM_001",
        name="Nominal Flight — All Systems Green",
        category="ENGINE",
        severity="ADVISORY",
        description="Tüm sistemler nominal. Baz durum testi — hiç uyarı olmamalı.",
        inject={
            "E1_TRQ": 78.0, "E2_TRQ": 77.0,
            "E1_OILP": 78.0, "E2_OILP": 79.0,
            "FUEL_L": 380.0, "FUEL_R": 370.0, "FUEL_TOT": 750.0, "FUEL_BAL": 10.0,
            "ELEC_GEN1V": 28.2, "ELEC_GEN2V": 28.1,
            "HYD_A_P": 3000.0, "HYD_B_P": 2950.0,
        },
        expected_mc_state="OFF",
        ai_vision_prompt=(
            "Bu ekranda: "
            "1) WCA paneli yeşil renkte mi (SYSTEM STATUS - MONITORING ACTIVE)? "
            "2) Master Caution 'OFF' yazıyor mu? "
            "3) Tüm göstergeler yeşil/nominal renkte mi? "
            "Her soru için EVET/HAYIR ile yanıtla."
        ),
        time_advance_secs=0,
        tick_count=1,
    ),
]


def get_scenario(scenario_id: str) -> Optional[FaultScenario]:
    """ID ile senaryo bul."""
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None


def get_scenarios_by_category(category: str) -> List[FaultScenario]:
    """Kategoriye göre filtrele."""
    return [s for s in SCENARIOS if s.category == category]


def get_scenarios_by_severity(severity: str) -> List[FaultScenario]:
    """Kritiklik düzeyine göre filtrele."""
    return [s for s in SCENARIOS if s.severity == severity]


CATEGORY_COLORS = {
    "ENGINE":        "#FF6B35",
    "FUEL":          "#FFD700",
    "ELECTRICAL":    "#00BFFF",
    "ENVIRONMENTAL": "#7CFC00",
    "HYDRAULIC":     "#FF69B4",
    "ROTOR":         "#DA70D6",
}

import random

# Senaryo sırasını rastgele karıştırmayı ve hata sayısını kısıtlamayı kaldırıyoruz.
# Kullanıcı talebi doğrultusunda 34 senaryonun tamamı çalıştırılacaktır.
# random.shuffle(SCENARIOS)
# num_scenarios_to_run = random.randint(15, 25)
# SCENARIOS = SCENARIOS[:num_scenarios_to_run]

# İnjekte edilen sayısal değerlere küçük ve güvenli rastgele sapmalar ekle
for sc in SCENARIOS:
    for k, v in sc.inject.items():
        if v == 0.0:
            sc.inject[k] = round(random.uniform(0.1, 0.9), 1)
        elif v == 100.0:
            sc.inject[k] = round(random.uniform(99.1, 99.9), 1)
        else:
            # -0.8 ile +0.8 arasında rastgele sapma
            # Sayı küçükse sapmayı daha da küçült ki limit sınırını geçmesin
            jitter = random.uniform(-0.8, 0.8) if abs(v) > 10 else random.uniform(-0.3, 0.3)
            sc.inject[k] = round(v + jitter, 1)
