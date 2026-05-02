
import os
import time
import random
import threading
from collections import deque
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
from langchain_core.runnables import RunnableLambda

class APIKeyManager:
    """
    API Anahtarlarını yöneten, 429 alanları karantinaya alan 
    ve arka planda iyileşenleri tekrar havuza katan yönetici sınıf.
    """
    def _get_key_suffix(self, key):
        """Key'in son 5 karakterini güvenli bir şekilde döndürür."""
        if not key: return "???"
        # SecretStr objesi ise (LangChain bazen sarmalar)
        if hasattr(key, "get_secret_value"):
            return key.get_secret_value()[-5:]
        return str(key)[-5:]

    def __init__(self):
        self.active_keys = deque()
        self.cooldown_keys = [] # (key, timestamp) tutmaz, thread ile kontrol edilir
        self.lock = threading.Lock()
        self.is_checking = False
        
        # 1. Ortam değişkenlerinden KEY'leri topla
        # GOOGLE_API_KEY, GOOGLE_API_KEY_2, ... şeklinde arar.
        self._load_keys()
        
        # 2. İyileşme kontrolü yapan thread'i başlat
        self.recovery_thread = threading.Thread(target=self._recovery_worker, daemon=True)
        self.recovery_thread.start()

    def _load_keys(self):
        # Keyleri 1'den 10'a kadar ara (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2...)
        for i in range(1, 11):
            key = os.environ.get(f"GOOGLE_API_KEY_{i}")
            if key:
                self.active_keys.append(key)
        
        if self.active_keys:
            first_key = self.active_keys[0]
            print(f"🔑 Key Manager Başlatıldı: {len(self.active_keys)} adet aktif anahtar var. (İlk Key: ...{self._get_key_suffix(first_key)})")
        else:
            print("⚠️ Key Manager Başlatıldı: HİÇ ANAHTAR BULUNAMADI!")

    def get_current_key(self):
        with self.lock:
            if not self.active_keys:
                raise Exception("❌ TÜM API ANAHTARLARI TÜKENDİ! Lütfen bekleyin veya yeni key ekleyin.")
            return self.active_keys[0] # Sıradakini döndür (rotate etmeden, hata alana kadar bunu kullan)

    def mark_as_exhausted(self, key):
        # Eğer SecretStr objesi ise (LangChain'den geliyorsa) gerçek string değerini al
        if hasattr(key, "get_secret_value"):
            key = key.get_secret_value()
            
        with self.lock:
            if key in self.active_keys:
                print(f"⚠️ Key Kotası Doldu (429): ...{self._get_key_suffix(key)} -> Karantinaya alınıyor.")
                self.active_keys.remove(key)
                self.cooldown_keys.append(key)
                print(f"🔄 Kalan Aktif Key Sayısı: {len(self.active_keys)}")
            else:
                # Debug için: Eğer key bulunamazsa nedenini anlamaya çalışalım
                print(f"DEBUG: Key {self._get_key_suffix(key)} aktif listede bulunamadı. Mevcut: {len(self.active_keys)}")

    def _recovery_worker(self):
        """
        Arka planda çalışan 'Doktor'. Karantinadaki keyleri test eder.
        """
        while True:
            time.sleep(300) # Her 5 dakikada bir kontrol et
            
            with self.lock:
                keys_to_check = list(self.cooldown_keys)
            
            if not keys_to_check:
                continue
                
            for key in keys_to_check:
                if self._test_key(key):
                    with self.lock:
                        if key in self.cooldown_keys:
                            self.cooldown_keys.remove(key)
                            self.active_keys.append(key)
                            print(f"✅ Key İyileşti ve Havuza Döndü: {key[:10]}...")

    def _test_key(self, key):
        """Key'in çalışıp çalışmadığını basit bir istek ile test eder."""
        try:
            # Çok ucuz bir modelle test et (gemini-1.5-flash)
            test_llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=key)
            test_llm.invoke("Hi")
            return True
        except Exception:
            return False

# Global Yönetici Nesnesi
key_manager = APIKeyManager()

class RobustGeminiModel:
    """
    LangChain model arayüzünü taklit eden, kendi içinde retry ve rotasyon mekanizması olan sarmalayıcı (wrapper).
    """
    def __init__(self, model_name="models/gemini-3-flash-preview", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def _get_current_llm(self):
        api_key = key_manager.get_current_key()
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key,
            # Retry'ı kapatıyoruz çünkü biz yöneteceğiz
            max_retries=0 
        )

    def invoke(self, *args, **kwargs):
        """
        İsteği yapar, hata olursa key değiştirip tekrar dener.
        """
        max_attempts = 10 # Toplam deneme hakkı (Key sayısı kadar veya daha fazla olabilir)
        
        for attempt in range(max_attempts):
            try:
                llm = self._get_current_llm()
                return llm.invoke(*args, **kwargs)
            
            except Exception as e:
                error_msg = str(e).upper()
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    exhausted_key = llm.google_api_key
                    key_manager.mark_as_exhausted(exhausted_key)
                    
                    try:
                        new_key = key_manager.get_current_key()
                        print(f"🔄 (String Match) 429 Tespit Edildi. ...{key_manager._get_key_suffix(exhausted_key)} çıkarıldı, yeni key: ...{key_manager._get_key_suffix(new_key)}")
                    except:
                        print(f"🔄 (String Match) 429 Tespit Edildi. ...{key_manager._get_key_suffix(exhausted_key)} çıkarıldı, BAŞKA KEY KALMADI!")
                    
                    time.sleep(1)
                    continue

                # Diğer sunucu hataları (503/500/Disconnect)
                if any(x in error_msg for x in ["503", "UNAVAILABLE", "500", "INTERNAL", "DISCONNECTED", "REMOTE ERROR", "CONNECTION"]):
                    wait_time = 20 * (attempt + 1)
                    print(f"⚠️ Google Sunucu Hatası ({error_msg[:30]}). {wait_time} sn bekleniyor...")
                    time.sleep(wait_time)
                    continue

                # Diğer beklenmeyen hatalarda (örn: prompt çok uzun) döngüyü kırmak lazım
                print(f"❌ Beklenmeyen Hata: {e}")
                raise e
        
        raise Exception("Maksimum deneme sayısına ulaşıldı, tüm keyler tükenmiş olabilir.")

    def with_structured_output(self, schema):
        """
        Structured output için özel wrapper.
        LangChain'in with_structured_output metodu yeni bir 'Runnable' döndürür.
        Biz de bu Runnable'ın invoke metodunu sarmalamalıyız.
        """
        def _structured_invoke(input_data, config=None, **kwargs):
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    llm = self._get_current_llm()
                    structured_llm = llm.with_structured_output(schema)
                    return structured_llm.invoke(input_data, config, **kwargs)
                
                except Exception as e:
                    error_msg = str(e).upper()
                    
                    # 429/RESOURCE_EXHAUSTED kontrolü (Hem tip hem mesaj bazlı)
                    if isinstance(e, ResourceExhausted) or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        exhausted_key = llm.google_api_key
                        key_manager.mark_as_exhausted(exhausted_key)
                        
                        try:
                            new_key = key_manager.get_current_key()
                            print(f"🔄 (Structured) 429/Kota Hatası. ...{key_manager._get_key_suffix(exhausted_key)} çıkarıldı, yeni key: ...{key_manager._get_key_suffix(new_key)}")
                        except:
                            print(f"🔄 (Structured) 429/Kota Hatası. ...{key_manager._get_key_suffix(exhausted_key)} çıkarıldı, BAŞKA KEY KALMADI!")
                        
                        time.sleep(1)
                        continue

                    if any(x in error_msg for x in ["503", "UNAVAILABLE", "500", "INTERNAL", "DISCONNECTED", "REMOTE ERROR", "CONNECTION"]):
                        wait_time = 20 * (attempt + 1)
                        print(f"⚠️ (Structured) Sunucu hatası ({error_msg[:30]}), {wait_time} sn bekleniyor...")
                        time.sleep(wait_time)
                        continue
                    
                    print(f"❌ (Structured) Beklenmeyen Hata: {e}")
                    raise e
            
            raise Exception("Maksimum deneme sayısına ulaşıldı.")
            
        return RunnableLambda(_structured_invoke)

def get_model():
    # Artık standart ChatGoogleGenerativeAI yerine bizim "Sağlam" sınıfımızı döndürüyoruz.
    # Bu sınıf 429 yerse kendi içinde key değiştirip devam edecek.
    return RobustGeminiModel(model_name="models/gemini-3-flash-preview", temperature=0)