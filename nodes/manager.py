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
    arxiv_queries: Optional[List[str]] = Field(description = "ArXiv akademik araması için İngilizce teknik terimler listesi (5 adet).")
    tavily_queries: Optional[List[str]] = Field(description = "Tavily web araması için İngilizce güncel internet verileri, bloglar ve haberler için arama terimleri listesi (5 adet).")

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
Görevin kullanıcının araştırma konusunu anlamak, gerekirse netleştirmek ve ArXiv ile Tavily aramaları için özelleştirilmiş arama terimleri oluşturmak.

ADIM 1 — KONU NETLEŞTİRME:
- Kullanıcı çok genel veya belirsiz bir konu verdiyse (örn: "yapay zeka"), 'is_clear' = False yap ve kısa, net bir soru sor.
- Kullanıcı makul düzeyde spesifik bir konu verdiyse (örn: "transformer mimarilerinde dikkat mekanizmaları"), bunu KABUL ET ve 'is_clear' = True yap.

ADIM 2 — ARAMA TERİMLERİ OLUŞTURMA:
Arama terimleri İNGİLİZCE olmalıdır. İki farklı kaynak için farklı stratejiler uygula:

1. ArXiv Araması (arxiv_queries - 5 adet):
- Akademik ve teknik literatüre odaklan.
- Piramit stratejisi: Geniş alan, spesifik alt alan, teknik detay, farklı uygulama ve survey/review terimleri üret.
- Örn: "large language models", "transformer attention optimization", "efficient attention mechanisms survey".

2. Tavily Web Araması (tavily_queries - 5 adet):
- Güncel internet verileri, bloglar, haberler, endüstri raporları ve genel bilgiye odaklan.
- Akademik olmayan ama konuyu destekleyen güncel gelişmeleri, popüler tartışmaları veya uygulama örneklerini hedefle.
- Örn: "latest LLM breakthroughs 2024", "AI industry trends blog", "commercial applications of transformers".

ARAMA TERİMİ KURALLARI:
- Doğal İngilizce ifadeler kullan (2-5 kelime).
- Cümle kurma, sadece anahtar kelime öbekleri yaz.
- Kısaltmalar (LLM, NLP vb.) kullanabilirsin.

ÖZEL DURUM — REVİZYON:
Eğer filterer_feedback varsa, önceki terimleri tekrarlama ve geri bildirimi dikkate alarak farklı açılar/terminolojiler dene.
- ArXiv için: Daha farklı teknik terimler veya komşu disiplinler.
- Tavily için: Daha güncel olaylar veya farklı sektör raporları."""

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