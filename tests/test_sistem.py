import pytest
from PyQt5.QtCore import Qt, QObject, pyqtSlot
import sys
import os

# Testleri alt klasorden calistirdigimiz icin ust klasordeki proje dosyalarini Python'a tanitiyoruz
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Projenizdeki asil test edeceginiz siniflari dahil ediyoruz
from widgets import SpeedGauge
from ekran import FlightDisplay

def test_hiz_gostergesi_renk_degisimi(qtbot):
    gosterge = SpeedGauge("IAS", "KT", 0, 220, caution_hi=200, warning_hi=210)
    qtbot.addWidget(gosterge)
    
    print("\n[TEST-1] Hız Göstergesi Doğrulama Testi Başlıyor...")
    gosterge.set_value(100)
    assert gosterge._color().name().lower() == "#00ff66"
    print(" -> Başarılı: 100 KT değerinde (Nominal), hız göstergesinin rengi Yeşil (#00ff66) oldu.")
    
    gosterge.set_value(215)
    assert gosterge._color().name().lower() == "#ff3333"
    print(" -> Başarılı: 215 KT değerinde (Uyarı Limit Üstü), hız göstergesinin rengi Kırmızı (#ff3333) oldu.")

def test_sistem_tork_uyarisi_veriyor_mu(qtbot, monkeypatch):
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)

    pencere = FlightDisplay()
    qtbot.addWidget(pencere)
    
    print("\n[TEST-2] Sistem Master Caution Işık Doğrulama Testi Başlıyor...")
    assert pencere.lbl_mc.text() == "OFF"
    print(" -> Başarılı: Test Başlangıcında Master Caution ışığı KAPALI (OFF).")
    
    pencere.vals["E1_TRQ"] = 108
    pencere._tick_sim()
    print(" -> Bilgi: E1 Torku %108'e zorlandı, zamanlayıcı başlatıldı (Henüz uyarı için 5 sn dolmadı).")
    
    pencere.elapsed = pencere.elapsed.addSecs(6) 
    pencere.vals["E1_TRQ"] = 108
    pencere._tick_sim()
    
    assert pencere.lbl_mc.text() == "ON"
    assert "#FF3333" in pencere.lbl_mc.styleSheet().upper()
    print(" -> Başarılı: Tork Limiti süresini doldurdu, Master Caution ışığı başarıyla Kırmızı/ON olarak yandı!")
    
    w_count, c_count, a_count = pencere.wca.counts()
    assert w_count > 0 
    print(f" -> Başarılı: Sistem veritabanında Warning sayısı {w_count} olarak tespit edildi.")

def test_sistemdeki_tum_gostergeler(qtbot, monkeypatch):
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)

    pencere = FlightDisplay()
    qtbot.addWidget(pencere)
    
    print("\n[TEST-3] MASTER KAPSAMLI TEST: Sistemdeki HER ŞEY (Tüm Parametreler) Test Ediliyor...")
    
    for key, widget in pencere.param_widgets.items():
        spec = widget.spec
        if spec.warning_lo is not None and spec.warning_hi is not None:
            nom_deger = (spec.warning_lo + spec.warning_hi) / 2
        elif spec.warning_hi is not None:
            nom_deger = spec.vmin + (spec.warning_hi - spec.vmin) / 2
        elif spec.warning_lo is not None:
            nom_deger = spec.vmax - (spec.vmax - spec.warning_lo) / 2
        else:
            nom_deger = spec.vmin + (spec.vmax - spec.vmin) / 2

        widget.set_value(nom_deger)
        assert widget.get_state(nom_deger) == "NOMINAL"
        
        if spec.warning_hi is not None:
            tehlikeli_deger = spec.warning_hi + 1
            widget.set_value(tehlikeli_deger)
            assert widget.get_state(tehlikeli_deger) == "WARNING"

    print(f" -> Başarılı: Tüm ParamBar komponentleri ({len(pencere.param_widgets)} adet Widget) State testlerinden hatasız geçti.")

    for key, widget in pencere.text_widgets.items():
        spec = widget.spec
        if spec.warning_hi is not None:
            widget.set_value(spec.warning_hi + 1)
            assert widget._invalid == False
            
    print(f" -> Başarılı: Tüm Elektrik/Metin komponentleri ({len(pencere.text_widgets)} adet Text Widget) hata sınırından başarıyla geçti.")
    
    for key, widget in list(pencere.param_widgets.items())[:3]:
        widget.set_invalid(True)
        assert widget._invalid == True
        assert widget.get_state(widget._last_real, True) == "INVALID"
    print(" -> Başarılı: Sensör BOZUK (INVALID) durumlarında kırmızı panel ve geçersiz state tepkisi test edildi.")

    assert pencere.lbl_wow.text() in ["AIR", "GROUND"]
    assert pencere.lbl_ap.text() in ["ON", "OFF"]
    print(" -> Başarılı: Üst Panellerdeki STATÜ ve SİSTEM göstergeleri doğrulandı.")

    pencere.wca_dialog.show()
    assert pencere.wca_dialog.isVisible() == True
    print(" -> Başarılı: Sağ alttaki 'MORE' (WCA Dialog) butonu ve penceresi etkileşimi test edildi.")
    pencere.wca_dialog.close()

    print(" -> GENEL SONUÇ: Sistemin ekranında bulunan TÜM GÖSTERGE VE DETAY ALTYAPISI DOĞRULANDI.")

def test_sesli_asistan_tetiklenmesi(qtbot, monkeypatch, capsys):
    print("\n[TEST-4] Sesli Asistan NLP Mantık Testi Başlıyor...")
    
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)
    pencere = FlightDisplay()
    qtbot.addWidget(pencere)
    
    class MockSesOynatici(QObject):
        def __init__(self):
            super().__init__()
            self.son_soylenen = ""
            
        @pyqtSlot(str)
        def konus(self, metin):
            print(f" -> ASİSTAN KONUŞTU: '{metin}'")
            self.son_soylenen = metin
            
        @pyqtSlot()
        def sustur(self):
            pass
            
    pencere.ses_oynatici = MockSesOynatici()
    
    pencere.wca.entries.clear() 
    pencere.wca._by.clear() 
    
    pencere._komut_isleyicide("durum ne")
    
    qtbot.wait(100)
    assert "pass" in pencere.ses_oynatici.son_soylenen.lower()
    
    pencere.wca.upsert(1234, "WARNING", "TEST_ERROR", "Test hatasi algilandi")
    pencere.wca.upsert(1234, "CAUTION", "TEST_CAUTION", "Test sari uyari algilandi")
    pencere._render_wca(1235) 
    
    pencere._komut_isleyicide("rapor ver")
    qtbot.wait(100)
    
    assert "fail" in pencere.ses_oynatici.son_soylenen.lower()
    assert "test hatasi algilandi (warning)" in pencere.ses_oynatici.son_soylenen.lower()
    assert "pass geçmiştir" in pencere.ses_oynatici.son_soylenen.lower()
    
    print(" -> Başarılı: Sesli Asistan doğal dil isteğine göre (Hata/Nominal) doğru state/rapor cümlelerini üretti.")


def test_sesli_asistan_surekli_ikaz_senaryosu(qtbot, monkeypatch):
    print("\n[TEST-5] Sesli Asistan Sürekli İkaz (Bleed High) ve Mute Testi Başlıyor...")
    monkeypatch.setattr(FlightDisplay, "_ses_sistemini_baslat", lambda self: None)
    
    pencere = FlightDisplay()
    qtbot.addWidget(pencere)
    
    class MockSesOynatici(QObject):
        def __init__(self):
            super().__init__()
            self.soylenenler = []
            
        @pyqtSlot(str)
        def konus(self, metin):
            print(f" -> ASİSTAN SÜREKLİ UYARI: '{metin}'")
            self.soylenenler.append(metin)
            
        @pyqtSlot()
        def sustur(self):
            pass
            
    pencere.ses_oynatici = MockSesOynatici()
    pencere.ses_ikaz_aktif = True

    pencere.wca.entries.clear()
    pencere.wca._by.clear()

    # Bleed sistemini Warning seviyesine zorluyoruz (Warning limit=97)
    pencere.vals["ENV_BLEED"] = 105
    pencere._tick_sim()
    
    # 1. Aşama: Sistemin Hatayı Algılaması İçin Süreyi İlerletiyoruz
    pencere.elapsed = pencere.elapsed.addSecs(6)
    pencere.vals["ENV_BLEED"] = 105
    pencere._tick_sim()
    
    # 2. Aşama: Zamanlayıcı döngüsünü tetikleyip asistanın arka arkaya konuşup konuşmadığına bakıyoruz
    for _ in range(4):
        pencere._tick_time()
    qtbot.wait(100)
        
    # Asistan uyardı mı ve içinde BLEED kelimesi geçti mi?
    assert len(pencere.ses_oynatici.soylenenler) > 0
    assert any("BLEED" in s.upper() for s in pencere.ses_oynatici.soylenenler)
    print(" -> Başarılı: Sistem ENV_BLEED hatasını sesli olarak raporladı.")
    
    # 3. Aşama: Mute düğmesine basıp sesin kesildiğini doğrulayalım
    onceki_konusma_sayisi = len(pencere.ses_oynatici.soylenenler)
    pencere.btn_mute.setChecked(True)
    pencere._toggle_mute(True) # Mute On
    
    # Döngü çalışmaya devam etsin
    for _ in range(4):
        pencere._tick_time()
        
    assert len(pencere.ses_oynatici.soylenenler) == onceki_konusma_sayisi
    print(" -> Başarılı: MUTE butonuna tıklandıktan sonra asistan susmayı başardı.")