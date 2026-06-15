import threading
import speech_recognition as sr
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot


import subprocess
import sys

class SesOynatici(QObject):
    def __init__(self):
        super().__init__()
        self.p = None

    @pyqtSlot(str)
    def konus(self, metin):
        # Önceki konuşma devam ediyorsa yenisini atla (overlap önleme)
        if self.p is not None and self.p.poll() is None:
            return
            
        code = f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
for v in voices:
    name = v.name.lower()
    if 'zira' in name or 'samantha' in name or 'victoria' in name or 'karen' in name:
        engine.setProperty('voice', v.id)
        break
engine.say({repr(metin)})
engine.runAndWait()
"""
        self.p = subprocess.Popen([sys.executable, "-c", code])

    @pyqtSlot()
    def sustur(self):
        # Mute butonuna basıldığında aktif konuşmayı ANINDA keser
        if self.p is not None and self.p.poll() is None:
            self.p.terminate()
            self.p = None


class DinleyiciThread(QThread):
    ses_duyuldu = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recognizer = sr.Recognizer()
        self.calisiyor = True

    def run(self):
        try:
            import pyaudio
            if pyaudio.PyAudio().get_device_count() == 0:
                print("Mikrofon bulunamadi.")
                return
        except Exception:
            print("PyAudio yuklenemedi.")
            return

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Sesli Asistan Dinliyor (Rapor / Durum diyerek test edebilirsiniz)...")
                while self.calisiyor:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                        metin = self.recognizer.recognize_google(audio, language="en-US").lower()
                        print(f"Pilot: {metin}")
                        self.ses_duyuldu.emit(metin)
                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except Exception:
                        pass
        except Exception as e:
            print(f"[DinleyiciThread] Mikrofon hatasi: {e}")

    def durdur(self):
        self.calisiyor = False
        self.wait()