from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from state import ResearchState
from model import get_model

# LLM'den dönecek yapılandırılmış veri formatı (JSON garantisi)
class ResearchProtocol(BaseModel):
    is_clear: bool = Field(description = "Kullanıcının konusu yeterince net mi? (True/False)")
    clarification_question: Optional[str] = Field(description = "Eğer net değilse sorulacak soru.")
    final_topic: Optional[str] = Field(description = "Netleşen akademik araştırma başlığı.")
    search_queries: Optional[List[str]] = Field(description = "ArXiv ve web araması için İngilizce arama terimleri listesi (5 adet).")

def manager_node(state: ResearchState):
    """ Kullanıcı ile mülakat yapar ve araştırma protokolünü belirler. """
    model = get_model()
    structured_output = model.with_structured_output(ResearchProtocol) # modelden JSON alıyoruz

    # Geçmiş mesajları al
    messages = state.get("messages", [])

    # Eğer mesaj listesi boşsa konuyu HumanMessage'a çevir
    if not messages:
        user_topic = state.get("user_topic", "")
        if user_topic:
            messages = [HumanMessage(content = user_topic)]
        else:
            # ikisi de yoksa varsayılan mesaj
            messages = [HumanMessage(content = "Lütfen akademik bir araştırma konusu belirle.")] # varsayılan mesaj

    system_prompt = """Sen bir akademik araştırma projesi yöneticisisin. 
Görevin kullanıcının araştırma konusunu anlamak, gerekirse netleştirmek ve ArXiv + web araması için etkili arama terimleri oluşturmak.

ADIM 1 — KONU NETLEŞTİRME:
- Kullanıcı çok genel veya belirsiz bir konu verdiyse (örn: "yapay zeka"), 'is_clear' = False yap ve kısa, net bir soru sor.
- Kullanıcı makul düzeyde spesifik bir konu verdiyse (örn: "transformer mimarilerinde dikkat mekanizmaları" veya "LLM'lerde fine-tuning yöntemleri"), bunu KABUL ET ve 'is_clear' = True yap.
- Aşırı detay bekleme. Konu bir akademik araştırma raporu yazılabilecek düzeyde netse yeterlidir.

ADIM 2 — ARAMA TERİMLERİ OLUŞTURMA (search_queries):
Arama terimleri İNGİLİZCE olmalıdır. ArXiv'de etkili sonuç getirmesi için şu piramit stratejisini uygula:

5 adet arama terimi üret:
  1. GENİŞ TERİM: Konunun ana alanını kapsayan geniş bir terim (örn: "large language models")
  2. ORTA TERİM 1: Konunun spesifik alt alanı (örn: "transformer attention mechanisms")
  3. ORTA TERİM 2: Konuyla ilgili farklı bir açı veya uygulama alanı (örn: "efficient attention architectures")
  4. ODAKLI TERİM: Konunun teknik detayına inen bir terim (örn: "multi-head attention optimization")
  5. SURVEY/REVIEW TERİMİ: Konuyla ilgili derleme makale bulacak terim (örn: "attention mechanisms survey" veya "transformer architectures review")

ARAMA TERİMİ KURALLARI:
- Terimleri ArXiv arama motoruna uygun, doğal İngilizce ifadeler olarak yaz.
- Her terim 2-5 kelime arasında olsun.
- Çok uzun veya cümle şeklinde terimler YAZMA.
- Kısaltmalar kullanılabilir (LLM, GAN, RL, NLP vb.)

ÖZEL DURUM — REVİZYON:
Eğer filterer_feedback bilgisi varsa, bu önceki aramanın yeterli sonuç getirmediği anlamına gelir.
Bu durumda:
- Önceki terimleri TEKRARLAMA, tamamen farklı açılardan yaklaş.
- Daha dar terimler üretme, aksine biraz daha geniş veya alternatif bakış açıları dene.
- Filterer'ın geri bildirimini dikkate alarak yaklaşımını değiştir.
- Konuya komşu alanlardan veya farklı terminolojilerden faydalanabilirsin."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("user", "Filterer Geri Bildirimi (Eğer varsa): {filterer_feedback}")
    ])

    # Zinciri oluştur ve çalıştır
    chain = prompt | structured_output
    feedback = state.get("filterer_feedback", "Yok")
    response: ResearchProtocol = chain.invoke({
        "messages": messages,
        "filterer_feedback": feedback
    })

    # Konu netleşti, protokolü kaydet ve kullanıcıya bilgi ver.
    if response.is_clear: 
        return {
            "research_protocol": response.model_dump(),
            "messages": [SystemMessage(content = f"Konu netleşti: '{response.final_topic}'. Araştırma başlatılıyor... (ArXiv ve Web taranıyor)")],
            "final_topic": response.final_topic,
            "filterer_feedback": None  # Önceki feedback'i temizle
        }
    # Konu net değil, soru sor.
    else: 
        return {"messages": [SystemMessage(content = response.clarification_question)]}