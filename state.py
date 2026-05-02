from typing import TypedDict, List, Dict, Annotated, Any
from langchain_core.messages import BaseMessage
import operator

class ResearchState(TypedDict):

    # İletişim (Manager ile chat) 
    messages: Annotated[List[BaseMessage], operator.add] # Ajanlar ve kullanıcı arasındaki tüm mesaj geçmişi

    # Giriş ve protokol  
    user_topic: str # Kullanıcının ilk sorgusu
    final_topic: str # Manager'in belirlediği, araştırılacak final konu
    research_protocol: Dict[str, Any] # Manager tarafından netleştirilmiş araştırma planı
    report_title: str # Analist tarafından belirlenen raporun akademik başlığı

    # Ham veriler (Researcher node çıktıları)    
    arxiv_summaries: List[Dict] # Arcivden gelen 125 makalenin metadatası (özet, başlık, yıl vs.)
    web_data: str # Tavily'den gelen web arama sonuçları (birleştirilmiş metin)
    web_sources: List[Dict] # Tavily kaynak metadataları [{title, url}] — referans bölümü için
    
    # İşlenmiş veriler (Filterer node çıktıları)
    selected_paper_ids: List[str] # Seçilen 15 makalenin listesi
    full_paper_contents: Dict[str, str] # Markdowna çevrilmiş makalelerin tam metni (paper_id: markdown text)
    paper_images: Dict[str, List[str]] # Makalelerden ayıklanan görsellerin yolları
    is_complete: bool # Araştırma sürecinin bitip bitmediğini kontrol eden bayrak
    revision_number: int # Kaçıncı araştırma denemesi? (Max 3)
    is_relevant: bool # Filterer gelen makaleleri beğendi mi?

    # Analiz (Analyst node için hazırlık)
    trend_analysis: str # 110 özetin trend analizi raporu (Genişlik)
    deep_analysis: str # 15 tam makalenin teknik analizi (Derinlik)

    # Raporlama (Analyst node'un iç döngüsü)
    report_outline: List[str] # Analistin oluşturduğu rapor iskeleti (bölüm başlıkları)
    current_section_index: int # Şu an yazılmakta olan bölümün sırası (iç döngü kontrolü için)
    written_sections: Dict[str, str] # Her bölümün yazılmış hâli. (başlık: içerik)
    
    # Çıktı ve kontrol
    final_report: str # Birleştirilmiş nihai rapor
    feedback: str # Reviewerin rapor hakkındaki geri bildirimi
    filterer_feedback: str # Filterer neden beğenmedi? (Manager'a not)