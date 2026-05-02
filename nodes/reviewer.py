from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import ResearchState
from model import get_model

class ReviewOutput(BaseModel):
    is_satisfactory: bool = Field(description = "Rapor, belirlenen standartlara ve plana uygun mu?")
    feedback: str = Field(description = "Eğer eksik varsa detaylı eleştiri, yoksa onay mesajı.")
    score: int = Field(description = "Rapora 100 üzerinden verilen akademik kalite notu.")

def reviewer_node(state: ResearchState):
    print("--- REVIEWER NODE: KALİTE KONTROL UZMANI ---")
    
    llm = get_model()
    structured_llm = llm.with_structured_output(ReviewOutput)
    
    # TÜM BAĞLAMI ÇEKİYORUZ
    user_topic = state.get("user_topic", "Bilinmiyor")
    final_topic = state.get("final_topic", "Bilinmiyor")
    protocol = state.get("research_protocol", {})
    report = state.get("final_report", "")
    
    # Bağlamı stringe çevirelim
    protocol_str = str(protocol.get("search_queries", []))
    
    system_prompt = """Sen bir akademik rapor kalite kontrol uzmanısın.
        Görevin, yazılan araştırma raporunu belirli kalite kriterlerine göre değerlendirmek.

        DEĞERLENDİRME KRİTERLERİ:

        1. KAYNAK BÜTÜNLÜĞÜ (En Kritik Kriter): 
           - Raporda metin içinde atıf yapılan HER kaynak (akademik makale, web sitesi) REFERANSLAR bölümünde listelenmeli.
           - 'Analiz', 'Analizler', 'Kaynak' gibi belirsiz atıflar var mı? Bunlar geçersiz atıflardır.
           - Kaynakçada olmayan bir makale/siteye atıf yapılmış mı? 
           - Uydurma bilgi veya halüsinasyon var mı?
        
        2. KONU UYUMU: Rapor, araştırma konusuna odaklı mı? Konu dağılmış mı yoksa tutarlı mı?
        
        3. AKADEMİK KALİTE: 
           - Akademik dil ve teknik derinlik yeterli mi?
           - Yüzeysel veya tekrarlayan ifadeler var mı?
           - Bölümler TAM MI? Cümle veya paragraf ortasında kesilmiş bölüm var mı?
        
        4. FORMAT VE YAPI: 
           - Bölüm başlıkları mantıklı bir sırayla mı? LaTeX doğru kullanılmış mı?
           - Rapor bütünlüklü ve okunabilir mi?
           - Eksik veya yarım kalmış bölüm var mı?

        PUANLAMA:
        - 80-100: Rapor yayınlanabilir kalitede.
        - 60-79: Küçük düzeltmelerle kabul edilebilir.
        - 40-59: Önemli eksiklikler var, revizyon gerekli.
        - 0-39: Ciddi sorunlar var, yeniden yazılmalı.
        
        KARAR:
        - is_satisfactory = True: Puan 60 veya üzeriyse.
        - is_satisfactory = False: Puan 60'ın altındaysa. Hangi bölümde ne sorun olduğunu detaylıca açıkla.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Kullanıcı İsteği: {user_topic}\nNetleşen Konu: {final_topic}\n\nRAPOR:\n{report}\n\nDEĞERLENDİR:")
    ])
    
    # Context sınırını koruyarak veriyoruz
    chain = prompt | structured_llm
    response = chain.invoke({
        "user_topic": user_topic,
        "final_topic": final_topic,
        "report": report
    })
    
    print(f"\n📊 KALİTE RAPORU")
    print(f"   Puan: {response.score}/100")
    print(f"   Karar: {'✅ GEÇTİ' if response.is_satisfactory else '❌ KALDI'}")
    print(f"   Feedback: {response.feedback}\n")

    return {
        "feedback": response.feedback,
        # İleride döngü kurarsak, Reviewer'ın notuna göre is_complete False yapılabilir.
        "is_complete": response.is_satisfactory 
    }