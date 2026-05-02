# nodes/filterer.py
import json
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tools.parsing_tools import process_paper, PDF_THREAD_COUNT
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from state import ResearchState
from model import get_model

# LLM Çıktısı için Yapı
class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description = "Makale havuzu araştırma konusuyla çalışılabilir düzeyde alakalı mı?")
    feedback: str = Field(description = "Eğer is_relevant=False ise Manager'a hangi tür terimlerle yeniden araması gerektiğine dair öneri.")
    selected_ids: list[str] = Field(description = "Seçilen makalelerin entry_id listesi (en az 5, en fazla 15).")

def filterer_node(state: ResearchState):
    print("--- FILTERER NODE: KALİTE KONTROLÜ YAPIYOR ---")
    
    model = get_model()
    structured_llm = model.with_structured_output(RelevanceCheck)
    
    # State'den verileri al
    summaries = state.get("arxiv_summaries", [])
    final_topic = state.get("final_topic", "")
    current_protocol = state.get("research_protocol", {})
    
    # LLM'e sadece başlık ve özetleri veriyoruz 
    context_text = "\n".join([f"ID: {s['entry_id']} - Title: {s['title']}\nSummary: {s['summary'][:500]}..." for s in summaries])
    
    system_prompt = f"""Sen bir akademik araştırma asistanısın. Görevin, ArXiv'den toplanan makale havuzundan araştırma konusuna en uygun olanları seçmek.

Araştırma Konusu: '{final_topic}'
Kullanılan Arama Terimleri: {current_protocol.get('search_queries')}

GÖREVİN:
Aşağıdaki makale listesini (başlık ve özet) incele ve araştırma konusuna fayda sağlayabilecek olanları seç.

SEÇİM KRİTERLERİ (Kapsayıcı Yaklaşım):
1. DOĞRUDAN ALAKALI: Makale araştırma konusunu doğrudan ele alıyorsa → KESİNLİKLE SEÇ.
2. DOLAYLI ALAKALI: Makale konuyla ilgili bir yöntem, teknik, karşılaştırma veya temel kavram sunuyorsa → SEÇ. Araştırma raporuna arka plan, bağlam veya kıyaslama olarak katkı sağlayabilir.
3. UZAK AMA FAYDALI: Makale konunun daha geniş alanıyla ilgiliyse ve raporda genel çerçeve çizmek için kullanılabilirse → DEĞERLENDİR ve mümkünse dahil et.
4. ALAKASIZ: Makale konuyla hiçbir şekilde ilişkilendirilemiyorsa → ELEME.

KARAR VERME:
- Hedef: En az 5, en fazla 15 makale seçmek.
- is_relevant = True yap: Eğer en az 5 adet doğrudan veya dolaylı alakalı makale varsa.
- is_relevant = False yap: SADECE havuzda 5'ten az kullanılabilir makale varsa. Bu durumda Manager'a farklı arama terimleri öner.

ÖNEMLİ UYARILAR:
- ArXiv araması keyword-based çalışır, bu yüzden bazı sonuçların kısmen alakasız olması NORMALDIR. Bu durum tek başına reddetme sebebi DEĞİLDİR.
- Mükemmel eşleşme arama. Araştırma raporuna herhangi bir şekilde katkı sağlayabilecek makaleleri dahil et.
- Şüphe durumunda DAHİL ET, dışarıda bırakma. Analyst node zaten en iyi kaynaklara odaklanacaktır."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "İncelemen gereken makale listesi:\n{paper_list}") 
    ])
    
    chain = prompt | structured_llm
    evaluation = chain.invoke({"paper_list": context_text})
        
    # DURUM 1: Makaleler kötü ise managera geri dön
    if not evaluation.is_relevant:
        print(f"Makaleler yetersiz bulundu. Sebep: {evaluation.feedback}")
        return {
            "filterer_feedback": evaluation.feedback,
            "revision_number": state.get("revision_number", 0) + 1,
        }

    # DURUM 2: Makaleler iyi ise işle ve devam et
    print(f"Makaleler onaylandı. {len(evaluation.selected_ids)} makale seçildi ve işleniyor...")
    
    # PDF URL'lerini hazırla
    pdf_tasks = []
    for paper_id in evaluation.selected_ids:
        pdf_url = None
        for summary in summaries:
            if summary["entry_id"] == paper_id:
                pdf_url = summary.get("pdf_url")
                break
        if pdf_url:
            pdf_tasks.append((paper_id, pdf_url))
        else:
            pdf_tasks.append((paper_id, None))
    
    # --- PARALEL PDF AYRIŞTIRMA (3 Thread) ---
    full_contents = {}
    
    def _process_single_paper(paper_id, pdf_url):
        """Tek bir makaleyi indirip ayrıştırır."""
        if not pdf_url:
            return paper_id, "PDF Linki Bulunamadı."
        print(f"🔍 [{paper_id[:15]}...] PDF ayrıştırılıyor...")
        content = process_paper(pdf_url)
        return paper_id, content
    
    print(f"⚡ {len(pdf_tasks)} makale {PDF_THREAD_COUNT} paralel thread ile ayrıştırılıyor...")
    
    with ThreadPoolExecutor(max_workers=PDF_THREAD_COUNT) as executor:
        futures = {
            executor.submit(_process_single_paper, pid, url): pid
            for pid, url in pdf_tasks
        }
        
        for future in as_completed(futures):
            try:
                paper_id, content = future.result()
                full_contents[paper_id] = content
                print(f"   ✅ [{paper_id[:15]}...] tamamlandı.")
            except Exception as e:
                paper_id = futures[future]
                full_contents[paper_id] = f"HATA: {e}"
                print(f"   ❌ [{paper_id[:15]}...] başarısız: {e}")
    
    print(f"📄 Toplam {len(full_contents)} makale ayrıştırıldı.")

    return {
        "selected_paper_ids": evaluation.selected_ids,
        "full_paper_contents": full_contents, # {Paper_id: Markdown text} formatında
        "filterer_feedback": None 
    }