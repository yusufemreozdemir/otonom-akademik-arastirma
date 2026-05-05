import os
import sys
from dotenv import load_dotenv

# .env yükle (Tüm proje importlarından önce olmalı)
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage

# Graph yapısını içe aktar
from graph import create_graph
from connection_handler import handle_connection_error, wait_for_connection, is_connection_error

def main():
    print("\n" + "="*60)
    print("🤖  OTONOM AKADEMİK ARAŞTIRMA ASİSTANI (v2.0 - Interactive)")
    print("="*60 + "\n")
    
    # 1. İlk Konuyu Al
    topic = input("📌 Lütfen araştırma konusunu girin: ")
    
    if not topic.strip():
        print("⚠️ Boş giriş yapıldı, program kapatılıyor.")
        return

    # 2. Başlangıç Durumu
    current_state = {
        "user_topic": topic,
        "revision_number": 0,
        "written_sections": {},
        "current_section_index": 0,
        "messages": [HumanMessage(content=topic)],
        "research_protocol": {"is_clear": False}, # Başlangıçta net değil varsayalım
        "final_report": None
    }

    # 3. Grafiği Hazırla
    try:
        app = create_graph()
    except Exception as e:
        print(f"❌ HATA: Graph oluşturulamadı: {e}")
        return

    # 4. ETKİLEŞİM DÖNGÜSÜ (Chat Loop)
    while True:
        print(f"\n⚙️  Sistem Çalışıyor... Lütfen bekleyin.")
        
        try:
            # Grafiği mevcut durumla çalıştır
            # invoke fonksiyonu, END'e ulaşana kadar çalışır.
            # Manager soru sorarsa END'e ulaşır, Rapor biterse yine END'e ulaşır.
            result = app.invoke(current_state)
            
            # Son durumu güncelle (Geçmiş mesajlar, yeni state vs. korunsun)
            current_state = result
            
        except Exception as e:
            # 🌐 İnternet kesintisi kontrolü — state korunarak kaldığı yerden devam
            if is_connection_error(e):
                if wait_for_connection(context="Graph çalışması sırasında"):
                    # Bağlantı geldi → current_state korunuyor, döngü başa dönecek
                    print("🔄 Sistem kaldığı yerden devam ediyor...")
                    continue
                else:
                    # Kullanıcı iptal etti
                    print("❌ İşlem iptal edildi. Mevcut ilerleme korunuyor.")
                    break
            
            print(f"\n❌ KRİTİK HATA: {e}")
            break

        # --- KONTROL NOKTASI ---
        
        # 1. SENARYO: Rapor Hazır mı? (MUTLU SON)
        if result.get("final_report"):
            print("\n" + "="*60)
            print("✅ SÜREÇ BAŞARIYLA TAMAMLANDI!")
            
            save_report(result)
            break # Döngüden çık
            
        # 2. SENARYO: Konu Netleşmemiş (Manager Soru Soruyor)
        protocol = result.get("research_protocol", {})
        if not protocol.get("is_clear"):
            # Manager'ın son mesajını (sorusunu) bulalım
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"\n❓ MANAGER SORUYOR: {last_msg.content}")
            else:
                print("\n❓ MANAGER: Konuyu tam anlayamadım, detay verebilir misiniz?")
            
            # Kullanıcıdan yeni cevap al
            print("-" * 40)
            new_input = input("👤 Cevabınız (Çıkış için 'q'): ")
            
            if new_input.lower() == 'q':
                print("Program sonlandırıldı.")
                break
            
            # Yeni cevabı mesaj geçmişine ekle ve döngüyü tekrarlat
            # Not: state.py'de messages için 'operator.add' kullandığımız için
            # burada listeye append etmek yerine, invoke'a yeni mesaj listesi veriyoruz,
            # LangGraph onu mevcut listeye ekliyor. Ancak invoke'a tüm state'i verdiğimiz için
            # manuel ekleyip state'i güncellemek daha garantidir.
            
            updated_messages = list(messages)
            updated_messages.append(HumanMessage(content=new_input))
            current_state["messages"] = updated_messages
            
            # Döngü başa döner -> app.invoke(current_state) tekrar çalışır
            continue
            
        # 3. SENARYO: Beklenmedik Durum (Rapor yok ama Konu Net?)
        else:
            print("⚠️ Beklenmedik durum: Konu net görünüyor ama rapor oluşmadı.")
            break

def save_report(state):
    final_report = state.get("final_report")
    report_title = state.get("report_title", state.get("final_topic", "arastirma_raporu"))
    
    # Raporlar için klasör oluştur
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Dosya ismini temizle (başlıktan güvenli dosya adı üret)
    safe_title = "".join([c if c.isalnum() else "_" for c in report_title])[:60]
    filename = os.path.join(report_dir, f"{safe_title}.md")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"📄 Rapor kaydedildi: {os.path.abspath(filename)}")
        print(f"📝 Kalite Notu: {state.get('feedback', 'Belirtilmedi')}")
    except IOError as e:
        print(f"❌ Kaydetme hatası: {e}")
        print("Rapor konsola basılıyor:\n", final_report)

if __name__ == "__main__":
    main()