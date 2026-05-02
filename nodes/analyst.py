import json
import re
import os
import base64
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import ResearchState
from model import get_model 

# --- ÇIKTI MODELLERİ (SCHEMA) ---

class ResearchAnalysis(BaseModel):
    report_title: str = Field(description="Raporun akademik başlığı. Kısa, açıklayıcı ve akademik formatta olmalı (Türkçe).")
    trend_analysis: str = Field(description="Literatürdeki genel eğilimler ve boşluklar.")
    deep_analysis: str = Field(description="Teknik derinlik ve metodolojik bulgular.")
    report_outline: List[str] = Field(description="Raporun bölüm başlıkları listesi (Sıralı ve Mantıksal).")

class SectionOutput(BaseModel):
    title: str = Field(description="Bölümün başlığı.")
    content: str = Field(description="Bölümün markdown içeriği (En az 400 kelime).")

# --- YARDIMCI FONKSİYONLAR ---

def _is_section_complete(content: str) -> bool:
    """Bir bölümün cümle ortasında kesilip kesilmediğini kontrol eder."""
    if not content or len(content.strip()) < 100:
        return False
    stripped = content.strip()
    if stripped[-1] in '.!?)»"\'':
        return True
    if stripped.endswith('$') or stripped.endswith('}'):
        return True
    last_part = stripped[-50:]
    if '.' in last_part and stripped.rstrip()[-1] in '.!?)»"\'$}':
        return True
    return False

def _fuzzy_match(title_a: str, title_b: str) -> bool:
    """İki bölüm başlığının benzer olup olmadığını kontrol eder."""
    a = title_a.lower().strip()[:40]
    b = title_b.lower().strip()[:40]
    return a in b or b in a

def _build_valid_citation_set(state: ResearchState) -> set:
    valid_citations = set()
    for s in state.get("arxiv_summaries", []):
        if s.get("entry_id") in state.get("selected_paper_ids", []):
            authors = s.get("authors", [])
            if authors:
                first_author_surname = authors[0].split()[-1] if authors[0] else ""
                if first_author_surname:
                    valid_citations.add(first_author_surname)
    for src in state.get("web_sources", []):
        title = src.get("title", "")
        if title:
            valid_citations.add(title)
            for word in title.split():
                if len(word) >= 2 and word[0].isupper():
                    valid_citations.add(word)
    return valid_citations

def _find_invalid_citations(report_text: str, valid_set: set) -> List[str]:
    author_citations = re.findall(r'(\w[\w\s]+?et al\.\s*,?\s*\d{4})', report_text)
    invalid = []
    for cite in author_citations:
        surname = cite.split('et al')[0].strip().split()[-1]
        if surname and surname not in valid_set:
            invalid.append(cite.strip())
    return list(set(invalid))

# --- PDF MULTIMODAL YARDIMCILARI ---

def _load_pdf_parts(state: ResearchState) -> list:
    """PDF dosyalarını Gemini multimodal mesaj parçalarına dönüştürür."""
    parts = []
    full_contents = state.get("full_paper_contents", {})
    summaries = state.get("arxiv_summaries", [])
    
    # paper_id → title eşlemesi
    title_map = {}
    for s in summaries:
        title_map[s.get("entry_id", "")] = s.get("title", "Başlıksız")
    
    loaded_count = 0
    for paper_id, pdf_path in full_contents.items():
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        
        title = title_map.get(paper_id, paper_id)
        
        try:
            with open(pdf_path, "rb") as f:
                pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            parts.append({"type": "text", "text": f"\n--- Makale: {title} (ID: {paper_id}) ---"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"}
            })
            loaded_count += 1
        except Exception as e:
            print(f"   ⚠️ PDF yüklenemedi ({paper_id[:15]}): {e}")
    
    print(f"   📎 {loaded_count} PDF doğrudan modele veriliyor.")
    return parts

# --- BÖLÜM TAMAMLAMA (KALDIĞI YERDEN DEVAM) ---

def _complete_incomplete_section(incomplete_content: str, section_title: str, topic: str) -> str:
    """Yarım kalan bölümü kaldığı yerden tamamlar. Sıfırdan yazmaz."""
    llm = get_model()
    
    tail = incomplete_content[-800:] if len(incomplete_content) > 800 else incomplete_content
    
    messages = [
        SystemMessage(content=f"""Sen uzman bir akademik yazarsın.
    Görevin: Yarım kalmış bir akademik bölümü KALDIGI YERDEN devam ettirerek tamamlamak.
    
    KURALLAR:
    1. Mevcut metni TEKRAR YAZMA. Sadece eksik olan kısmı yaz.
    2. Mevcut metnin son cümlesini anlamlı şekilde tamamla.
    3. Bir kapanış paragrafı ekleyerek bölümü sonlandır.
    4. Son cümle MUTLAKA nokta (.) ile bitmeli.
    5. Akademik dil ve LaTeX kullanımını koru.
    6. Bölüm başlığı: {section_title}
    7. Araştırma konusu: {topic}"""),
        HumanMessage(content=f"YARIM KALAN METNİN SONU:\n...{tail}\n\nMetni kaldığı yerden devam ettir ve tamamla. SADECE eksik kısmı yaz:")
    ]
    
    response = llm.invoke(messages)
    completion = response.content if hasattr(response, 'content') else str(response)
    combined = incomplete_content.rstrip() + " " + completion.strip()
    return combined

# --- ANA DÜĞÜM ---

def analyst_node(state: ResearchState):
    print("--- ANALYST NODE ÇALIŞIYOR (1x1 Mode) ---")
    
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    feedback = state.get("feedback", "")
    review_count = state.get("review_count", 0)
    failed_titles = state.get("failed_section_titles", [])
    
    # Reviewer'dan geri döndüyse — SADECE sorunlu bölümleri temizle
    if feedback and outline and current_index >= len(outline):
        written = state.get("written_sections", {}).copy()
        removed = []
        
        for failed in failed_titles:
            for otitle in outline:
                if _fuzzy_match(failed, otitle) and otitle in written:
                    del written[otitle]
                    removed.append(otitle)
                    break
        
        for title in outline:
            if title in written and not _is_section_complete(written[title]):
                del written[title]
                if title not in removed:
                    removed.append(title)
        
        if removed:
            print(f"🔄 Reviewer'dan geri dönüldü (Deneme {review_count + 1}/3). {len(removed)} bölüm yeniden yazılacak:")
            for r in removed:
                print(f"   - {r[:50]}")
        else:
            print(f"🔄 Reviewer'dan geri dönüldü (Deneme {review_count + 1}/3). Kaynak düzeltmesi yapılacak.")
        
        return {
            "written_sections": written,
            "current_section_index": 0,
            "is_complete": False,
            "feedback": "",  # Feedback'i temizle ki sonsuz döngüye girmesin
            "review_count": review_count + 1,
            "failed_section_titles": []
        }
    
    if not outline:
        return run_architect_mode(state)
    elif current_index < len(outline):
        target_title = outline[current_index]
        written = state.get("written_sections", {})
        
        if target_title in written and _is_section_complete(written[target_title]):
            print(f"⏭️ Bölüm {current_index+1}/{len(outline)} zaten tamamlanmış, atlanıyor: '{target_title[:40]}...'")
            return {
                "current_section_index": current_index + 1,
                "is_complete": False
            }
        
        if target_title in written and not _is_section_complete(written[target_title]):
            return run_completion_mode(state)
        
        return run_batch_writer(state)
    else:
        return run_editor_mode(state)

# --- ALT FONKSİYONLAR ---

def _get_allowed_citations(state: ResearchState) -> str:
    allowed = "İZİN VERİLEN ATIF FORMATLARI (Bunlar dışında HİÇBİR kaynağa/yazara atıf yapma!):\n"
    
    allowed += "[AKADEMİK MAKALELER]\n"
    selected_ids = state.get("selected_paper_ids", [])
    for s in state.get("arxiv_summaries", []):
        if s.get("entry_id") in selected_ids:
            authors = s.get("authors", [])
            first_author = authors[0] if authors else ""
            surname = first_author.split()[-1] if first_author else "Bilinmeyen"
            year = s.get("published", "2023")[:4]
            title = s.get("title", "Başlıksız")
            allowed += f"- ({surname} et al., {year}) | {title}\n"
            
    allowed += "\n[WEB KAYNAKLARI]\n"
    for src in state.get("web_sources", []):
        title = src.get("title", "Başlıksız")
        allowed += f"- ({title})\n"
        
    return allowed

def run_architect_mode(state: ResearchState):
    print("🧠 Mod: MİMAR (Analiz ve Planlama)")
    
    llm = get_model()
    structured_llm = llm.with_structured_output(ResearchAnalysis)
    
    topic = state.get("final_topic", state.get("user_topic"))
    summaries = state.get("arxiv_summaries", [])
    web_data = state.get("web_data", "")
    
    summary_text = "\n".join([f"- {s['title']}: {s['summary'][:300]}" for s in summaries])
    pdf_parts = _load_pdf_parts(state)
    
    system_prompt = f"""Sen deneyimli bir akademik araştırmacısın.
    Görevin: Verilen kaynakları analiz ederek kapsamlı bir akademik araştırma raporu planı oluşturmak.
    
    KURALLAR:
    1. KONU ODAKLI OL: Sadece '{topic}' konusuna odaklan. 
    2. AKADEMİK DİL: Ciddi, teknik ve nesnel bir dil kullan.
    3. LATEX KULLANIMI: Tüm teknik terimler ve formüller için LaTeX kullan.
    4. KAYNAK TABANLI: Analizini sadece sana verilen kaynaklara dayandır.
    
    ADIM A — GENİŞLİK ANALİZİ: Genel eğilimleri ve araştırma boşluklarını tespit et.
    ADIM B — DERİNLİK ANALİZİ: Teknik detayları ve bulguları analiz et.
    ADIM C — RAPOR İSKELETİ: Mantıksal bir sırayla 6-10 bölüm başlığı oluştur.
    """
    
    # Multimodal mesaj: metin + PDF dosyaları
    user_content = [
        {"type": "text", "text": f"Konu: {topic}\n\nWeb Verisi: {web_data[:30000]}\n\nÖzetler:\n{summary_text}\n\n--- AKADEMİK MAKALELER (PDF) ---"},
        *pdf_parts,
        {"type": "text", "text": "\n\nYukarıdaki kaynakları analiz et ve rapor planı oluştur."}
    ]
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = structured_llm.invoke(messages)
    
    print(f"✅ Planlama Tamamlandı: '{response.report_title}' — {len(response.report_outline)} bölüm belirlendi.")
    
    return {
        "report_title": response.report_title,
        "trend_analysis": response.trend_analysis,
        "deep_analysis": response.deep_analysis,
        "report_outline": response.report_outline,
        "written_sections": {},
        "current_section_index": 0,
        "is_complete": False
    }

def run_completion_mode(state: ResearchState):
    """Yarım kalan bölümü kaldığı yerden tamamlar — sıfırdan yazmaz."""
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    target_title = outline[current_index]
    topic = state.get("final_topic", state.get("user_topic"))
    existing_content = state.get("written_sections", {}).get(target_title, "")
    
    print(f"🔧 Mod: TAMAMLAYICI (Bölüm {current_index+1}/{len(outline)}: '{target_title[:40]}...')")
    
    completed = _complete_incomplete_section(existing_content, target_title, topic)
    
    if not _is_section_complete(completed):
        print(f"   ⚠️ İlk tamamlama yetmedi, bir deneme daha...")
        completed = _complete_incomplete_section(completed, target_title, topic)
    
    new_written = state.get("written_sections", {}).copy()
    new_written[target_title] = completed
    
    status = "✅ tamamlandı" if _is_section_complete(completed) else "⚠️ hala eksik, devam ediliyor"
    print(f"   -> Bölüm {status}: {target_title[:40]}...")
    
    return {
        "written_sections": new_written,
        "current_section_index": current_index + 1,
        "is_complete": False
    }

def run_batch_writer(state: ResearchState):
    """Yeni bir bölümü sıfırdan yazar — PDF'leri doğrudan modele vererek."""
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    target_title = outline[current_index]
    topic = state.get("final_topic", state.get("user_topic"))
    
    print(f"✍️ Mod: YAZAR (Bölüm {current_index+1}/{len(outline)}: '{target_title[:40]}...')")

    llm = get_model()
    structured_llm = llm.with_structured_output(SectionOutput)
    
    web_text = state.get("web_data", "")[:30000]
    pdf_parts = _load_pdf_parts(state)
    
    written_sections = state.get("written_sections", {})
    written_titles = list(written_sections.keys())
    
    # Geçmiş bağlam: O ana kadar yazılmış TÜM bölümlerin tam metnini al
    history_context = ""
    if current_index > 0:
        context_parts = []
        for i in range(0, current_index):
            prev_title = outline[i]
            prev_content = written_sections.get(prev_title, "")
            if prev_content:
                context_parts.append(f"--- ÖNCEKİ BÖLÜM: {prev_title} ---\n{prev_content}")
        if context_parts:
            history_context = "\n\n".join(context_parts)
            history_context = f"\nRAPORUN ŞİMDİYE KADAR YAZILAN KISMI (Akışı ve bütünlüğü buna göre ayarla):\n{history_context}\n"

    reviewer_feedback = state.get("feedback", "")
    feedback_instruction = ""
    if reviewer_feedback:
        feedback_instruction = f"""
    
    ÖNCEKİ İNCELEME GERİ BİLDİRİMİ (Bu sorunları DÜZELT):
    {reviewer_feedback}
    """
    
    allowed_citations = _get_allowed_citations(state)
    
    system_prompt = f"""Sen uzman bir akademik yazarsın.
    GÖREVİN: Aşağıdaki TEK bölüm başlığı için kapsamlı bir akademik bölüm yazmak.
    
    YAZILACAK BÖLÜM: {target_title}
    
    YAZIM KURALLARI:
    1. KONU SADAKATİ: Araştırma konusu '{topic}'.
    2. ÇIKTI FORMATI: TEK bir SectionOutput objesi döndür. title ve content alanlarını doldur.
    3. BAŞLIK: title alanına '{target_title}' yaz (DEĞİŞTİRME).
    4. KAYNAK SADAKATİ: Sadece sana verilen Akademik Makaleler (PDF) VE Web Kaynaklarını kullan.
    5. LATEX KULLANIMI: Formüller için LaTeX kullan. Firma/Model isimleri (Örn: GPT-4, Anthropic) için LaTeX KULLANMA, normal metin yaz. ÇOK ÖNEMLİ: Çıktın JSON olacağı için LaTeX yazarken ters bölü (\) işaretini ÇİFT kullan! (Örn: \text yerine \\text yazmalısın. Aksi halde \t tab olarak algılanır ve kelimeler bozulur).
    6. KESİN ATIF KURALI: YALNIZCA aşağıda listelenen "İZİN VERİLEN ATIF FORMATLARI"nı kullanabilirsin. PDF içindeki metinlerde geçen ancak bu listede olmayan BAŞKA YAZARLARA veya kaynaklara (örn. Kaplan, Chen vb.) KESİNLİKLE ATIF YAPMA. Halüsinasyon atıf uydurma.
    7. BÖLÜM UZUNLUĞU: En az 400 kelime, akademik ve detaylı.
    8. BÖLÜM BÜTÜNLÜĞÜ: Bölümü MUTLAKA tamamla. Cümle ortasında bırakma. SON CÜMLE MUTLAKA NOKTAYLA BİTMELİ.
    9. YASAK ATIFLAR: 'Analiz', 'Analizler', 'Araştırma', 'Kaynak', 'Değerlendirme', 'İnceleme', 'Tavily AI Summary' kelimeleri ASLA atıf olarak kullanılamaz.
    10. AKIŞ VE BAĞLAM: Sana verilen "GEÇMİŞ BAĞLAM"ı oku. Önceki bölümlerde anlatılanları tekrar etme, onların üzerine inşa et ve mantıksal bir geçiş sağla.
    
    {allowed_citations}
    
    {history_context}
    
    {feedback_instruction}
    """
    
    # Multimodal mesaj: metin + PDF dosyaları
    user_content = [
        {"type": "text", "text": f"Konu: {topic}\n\nYAZIM REHBERİ (atıf kaynağı olarak KULLANMA):\n{state.get('trend_analysis', '')}\n\nATIF YAPILABİLİR KAYNAKLAR — Web Siteleri:\n{web_text}\n\nATIF YAPILABİLİR KAYNAKLAR — Akademik Makaleler (PDF):"},
        *pdf_parts,
        {"type": "text", "text": f"\n\nLütfen '{target_title}' başlıklı bölümü yaz:"}
    ]
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = structured_llm.invoke(messages)
    final_content = response.content
    
    # Eksikse → kaldığı yerden tamamla
    if not _is_section_complete(final_content):
        print(f"   ⚠️ Bölüm eksik tespit edildi, kaldığı yerden tamamlanıyor...")
        final_content = _complete_incomplete_section(final_content, target_title, topic)
        
        if _is_section_complete(final_content):
            print(f"   ✅ Bölüm tamamlama modunda bitirildi.")
        else:
            print(f"   ⚠️ Bölüm tamamlanamadı, mevcut haliyle kaydediliyor.")
    
    new_written = state.get("written_sections", {}).copy()
    new_written[target_title] = final_content
    print(f"   -> Bölüm yazıldı: {target_title[:40]}...")
    
    return {
        "written_sections": new_written,
        "current_section_index": current_index + 1,
        "is_complete": False
    }

def run_editor_mode(state: ResearchState):
    print("🏁 Mod: EDİTÖR (Rapor Birleştiriliyor - API Çağrısı YOK)")
    
    outline = state.get("report_outline", [])
    written_sections = state.get("written_sections", {})
    topic = state.get("final_topic", state.get("user_topic"))
    
    report_title = state.get("report_title", topic)
    full_report = f"# {report_title}\n\n"
    
    incomplete_sections = []
    for title in outline:
        content = written_sections.get(title)
        
        if not content:
            for k, v in written_sections.items():
                if k in title or title in k:
                    content = v
                    break
        
        if content:
            # LaTeX JSON kaçış hatalarını düzelt (\text -> <tab>ext sorununu çözer)
            content = content.replace("\text{", "\\text{")
            content = content.replace("$\t", "$\\t")
            content = content.replace("$ ext{", "$\\text{")
            
            if not _is_section_complete(content):
                incomplete_sections.append(title)
                print(f"   ⚠️ Eksik bölüm tespit edildi: {title[:40]}...")
            full_report += f"## {title}\n\n{content}\n\n"
        else:
            full_report += f"## {title}\n\n*(Bu bölüm oluşturulurken teknik bir aksaklık yaşandı)*\n\n"
            incomplete_sections.append(title)

    valid_citations = _build_valid_citation_set(state)
    invalid_citations = _find_invalid_citations(full_report, valid_citations)
    
    if invalid_citations:
        print(f"   ⚠️ Kaynakçada bulunmayan {len(invalid_citations)} atıf tespit edildi: {invalid_citations[:5]}")
    if incomplete_sections:
        print(f"   ⚠️ {len(incomplete_sections)} eksik bölüm tespit edildi: {[t[:30] for t in incomplete_sections]}")

    # REFERANSLAR
    full_report += "## REFERANSLAR\n\n"
    
    full_report += "### Akademik Kaynaklar\n"
    selected_ids = state.get("selected_paper_ids", [])
    academic_added = False
    
    for s in state.get("arxiv_summaries", []):
        if s['entry_id'] in selected_ids:
            authors = s.get('authors', ['Bilinmeyen Yazar'])
            first_author = authors[0] if authors else 'Bilinmeyen Yazar'
            published = s.get('published', 'Tarih Bilinmiyor')
            entry_id = s.get('entry_id', '')
            full_report += f"- {first_author} et al. ({published[:4]}). **{s['title']}**. arXiv: {entry_id}\n"
            academic_added = True
    
    if not academic_added:
        full_report += "- *Akademik kaynak bulunamadı.*\n"
    
    full_report += "\n### Web Kaynakları\n"
    web_sources = state.get("web_sources", [])
    web_added = False
    
    for src in web_sources:
        url = src.get("url", "")
        title = src.get("title", "Başlıksız")
        if url:
            full_report += f"- {title}. Erişim: {url}\n"
            web_added = True
            
    if not web_added:
        full_report += "- *Web kaynağı bulunamadı.*\n"

    print(f"✅ Rapor Hazır.")
    
    return {
        "final_report": full_report,
        "is_complete": True
    }