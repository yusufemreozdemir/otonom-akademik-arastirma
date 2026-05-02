import json
import re
import math
from typing import List
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

def _escape_braces(text):
    if isinstance(text, str):
        return text.replace("{", "{{").replace("}", "}}")
    elif isinstance(text, list):
        return [_escape_braces(item) for item in text]
    return str(text).replace("{", "{{").replace("}", "}}")

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

# --- BÖLÜM TAMAMLAMA (KALDIĞI YERDEN DEVAM) ---

def _complete_incomplete_section(incomplete_content: str, section_title: str, topic: str) -> str:
    """Yarım kalan bölümü kaldığı yerden tamamlar. Sıfırdan yazmaz."""
    llm = get_model()
    
    # Son 800 karakteri bağlam olarak ver
    tail = incomplete_content[-800:] if len(incomplete_content) > 800 else incomplete_content
    safe_title = _escape_braces(section_title)
    safe_topic = _escape_braces(topic)
    safe_tail = _escape_braces(tail)
    
    system_prompt = f"""Sen uzman bir akademik yazarsın.
    Görevin: Yarım kalmış bir akademik bölümü KALDIGI YERDEN devam ettirerek tamamlamak.
    
    KURALLAR:
    1. Mevcut metni TEKRAR YAZMA. Sadece eksik olan kısmı yaz.
    2. Mevcut metnin son cümlesini anlamlı şekilde tamamla.
    3. Bir kapanış paragrafı ekleyerek bölümü sonlandır.
    4. Son cümle MUTLAKA nokta (.) ile bitmeli.
    5. Akademik dil ve LaTeX kullanımını koru.
    6. Bölüm başlığı: {safe_title}
    7. Araştırma konusu: {safe_topic}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "YARIM KALAN METNİN SONU:\n...{tail_text}\n\nMetni kaldığı yerden devam ettir ve tamamla. SADECE eksik kısmı yaz:")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"tail_text": tail})
    
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
        
        # 1. Reviewer'ın belirttiği bölümleri temizle (fuzzy match)
        for failed in failed_titles:
            for otitle in outline:
                if _fuzzy_match(failed, otitle) and otitle in written:
                    del written[otitle]
                    removed.append(otitle)
                    break
        
        # 2. Otomatik: Yarım kalan bölümleri de temizle
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
            "feedback": feedback,
            "review_count": review_count + 1,
            "failed_section_titles": []
        }
    
    # 1. ADIM: MİMAR
    if not outline:
        return run_architect_mode(state)
    
    # 2. ADIM: YAZAR
    elif current_index < len(outline):
        target_title = outline[current_index]
        written = state.get("written_sections", {})
        
        # Bu bölüm zaten yazılmış ve tamamsa → ATLA
        if target_title in written and _is_section_complete(written[target_title]):
            print(f"⏭️ Bölüm {current_index+1}/{len(outline)} zaten tamamlanmış, atlanıyor: '{target_title[:40]}...'")
            return {
                "current_section_index": current_index + 1,
                "is_complete": False
            }
        
        # Yarım kalmış bölüm varsa → TAMAMLAMA MODU
        if target_title in written and not _is_section_complete(written[target_title]):
            return run_completion_mode(state)
        
        # Yeni bölüm → YAZIM MODU
        return run_batch_writer(state)
        
    # 3. ADIM: EDİTÖR
    else:
        return run_editor_mode(state)

# --- ALT FONKSİYONLAR ---

def run_architect_mode(state: ResearchState):
    print("🧠 Mod: MİMAR (Analiz ve Planlama)")
    
    llm = get_model()
    structured_llm = llm.with_structured_output(ResearchAnalysis)
    
    topic = state.get("final_topic", state.get("user_topic"))
    summaries = state.get("arxiv_summaries", [])
    full_contents = state.get("full_paper_contents", {})
    web_data = state.get("web_data", "")
    
    summary_text = "\n".join([f"- {s['title']}: {s['summary'][:300]}" for s in summaries])
    full_text = "\n\n".join([f"--- PAPER ID: {pid} ---\n{str(content)[:25000]}" for pid, content in full_contents.items()])
    
    system_prompt = f"""Sen deneyimli bir akademik araştırmacısın.
    Görevin: Verilen kaynakları analiz ederek kapsamlı bir akademik araştırma raporu planı oluşturmak.
    
    KURALLAR:
    1. KONU ODAKLI OL: Sadece '{_escape_braces(topic)}' konusuna odaklan. 
    2. AKADEMİK DİL: Ciddi, teknik ve nesnel bir dil kullan.
    3. LATEX KULLANIMI: Tüm teknik terimler ve formüller için LaTeX kullan.
    4. KAYNAK TABANLI: Analizini sadece sana verilen kaynaklara dayandır.
    
    ADIM A — GENİŞLİK ANALİZİ: Genel eğilimleri ve araştırma boşluklarını tespit et.
    ADIM B — DERİNLİK ANALİZİ: Teknik detayları ve bulguları analiz et.
    ADIM C — RAPOR İSKELETİ: Mantıksal bir sırayla 6-10 bölüm başlığı oluştur.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Konu: {topic}\n\nWeb Verisi: {web_data}\n\nÖzetler:\n{summary_text}\n\nTam Metinler:\n{full_text}")
    ])
    
    chain = prompt | structured_llm
    response = chain.invoke({
        "topic": topic,
        "web_data": web_data,
        "summary_text": summary_text,
        "full_text": full_text
    })
    
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
    """Yeni bir bölümü sıfırdan yazar."""
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    target_title = outline[current_index]
    topic = state.get("final_topic", state.get("user_topic"))
    
    print(f"✍️ Mod: YAZAR (Bölüm {current_index+1}/{len(outline)}: '{target_title[:40]}...')")

    llm = get_model()
    structured_llm = llm.with_structured_output(SectionOutput)
    
    full_contents_text = str(state.get("full_paper_contents", {}))[:100000]
    web_text = state.get("web_data", "")[:30000]
    
    safe_title = _escape_braces(target_title)
    safe_topic = _escape_braces(topic)
    
    written_so_far = list(state.get("written_sections", {}).keys())
    prev_context = ""
    if written_so_far:
        prev_context = f"Şimdiye kadar yazılan bölümler: {_escape_braces(str(written_so_far))}. Tekrar yazmaya gerek yok."
    
    reviewer_feedback = state.get("feedback", "")
    feedback_instruction = ""
    if reviewer_feedback:
        safe_feedback = _escape_braces(reviewer_feedback)
        feedback_instruction = f"""
    
    ÖNCEKİ İNCELEME GERİ BİLDİRİMİ (Bu sorunları DÜZELT):
    {safe_feedback}
    """
    
    system_prompt = f"""Sen uzman bir akademik yazarsın.
    GÖREVİN: Aşağıdaki TEK bölüm başlığı için kapsamlı bir akademik bölüm yazmak.
    
    YAZILACAK BÖLÜM: {safe_title}
    
    YAZIM KURALLARI:
    1. KONU SADAKATİ: Araştırma konusu '{safe_topic}'.
    2. ÇIKTI FORMATI: TEK bir SectionOutput objesi döndür. title ve content alanlarını doldur.
    3. BAŞLIK: title alanına '{safe_title}' yaz (DEĞİŞTİRME).
    4. KAYNAK SADAKATİ: Sadece sana verilen Akademik Makaleler VE Web Kaynaklarını kullan.
    5. LATEX KULLANIMI: Teknik terimler ve formüller için LaTeX kullan.
    6. HALÜSİNASYON YASAĞI: Kaynakta olmayan bilgi veya atıf EKLEME. Sana verilmeyen yazarları cite etme.
    7. ATIF FORMATI:
       - Akademik makaleler için: Yazar et al. (Yıl) — SADECE aşağıdaki Akademik Kaynaklar bölümünde bulunan yazarları kullan.
       - Web kaynakları için: Site adı — SADECE aşağıdaki Web Kaynakları bölümünde bulunan site adlarını kullan.
    8. BÖLÜM UZUNLUĞU: En az 400 kelime, akademik ve detaylı.
    9. BÖLÜM BÜTÜNLÜĞÜ: Bölümü MUTLAKA tamamla. Cümle ortasında bırakma. Açılış, gelişme ve kapanış paragrafı olmalı.
       SON CÜMLE MUTLAKA NOKTAYLA BİTMELİ. Eksik cümle YASAK.
    10. YASAK ATIFLAR: 'Analiz', 'Analizler', 'Araştırma', 'Kaynak', 'Değerlendirme', 'İnceleme', 'Tavily AI Summary'
        kelimeleri ASLA atıf olarak kullanılamaz. Bunlar kaynak değildir.
    {prev_context}{feedback_instruction}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", """Konu: {topic}

YAZIM REHBERİ (atıf kaynağı olarak KULLANMA):
{trend}

ATIF YAPILABİLİR KAYNAKLAR — Akademik Makaleler:
{full_text}

ATIF YAPILABİLİR KAYNAKLAR — Web Siteleri:
{web_text}

Lütfen '{section_title}' başlıklı bölümü yaz:""")
    ])
    
    chain = prompt | structured_llm
    response = chain.invoke({
        "topic": topic,
        "trend": state.get('trend_analysis', ''),
        "full_text": full_contents_text,
        "web_text": web_text,
        "section_title": target_title
    })
    
    final_content = response.content
    
    # Eksikse → sıfırdan yazma, kaldığı yerden tamamla
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
    
    # 1. Outline sırasına göre birleştir
    incomplete_sections = []
    for title in outline:
        content = written_sections.get(title)
        
        if not content:
            for k, v in written_sections.items():
                if k in title or title in k:
                    content = v
                    break
        
        if content:
            if not _is_section_complete(content):
                incomplete_sections.append(title)
                print(f"   ⚠️ Eksik bölüm tespit edildi: {title[:40]}...")
            full_report += f"## {title}\n\n{content}\n\n"
        else:
            full_report += f"## {title}\n\n*(Bu bölüm oluşturulurken teknik bir aksaklık yaşandı)*\n\n"
            incomplete_sections.append(title)

    # 2. KAYNAK DOĞRULAMA
    valid_citations = _build_valid_citation_set(state)
    invalid_citations = _find_invalid_citations(full_report, valid_citations)
    
    if invalid_citations:
        print(f"   ⚠️ Kaynakçada bulunmayan {len(invalid_citations)} atıf tespit edildi: {invalid_citations[:5]}")
    if incomplete_sections:
        print(f"   ⚠️ {len(incomplete_sections)} eksik bölüm tespit edildi: {[t[:30] for t in incomplete_sections]}")

    # 3. REFERANSLAR — sadece metin içinde atıf yapılanları dahil et
    full_report += "## REFERANSLAR\n\n"
    
    # 3a. Akademik Kaynaklar
    full_report += "### Akademik Kaynaklar\n"
    seen_refs = set()
    selected_ids = state.get("selected_paper_ids", [])
    
    for s in state.get("arxiv_summaries", []):
        if s['entry_id'] in selected_ids and s['title'] not in seen_refs:
            authors = s.get('authors', ['Bilinmeyen Yazar'])
            first_author = authors[0] if authors else 'Bilinmeyen Yazar'
            surname = first_author.split()[-1] if first_author else ''
            published = s.get('published', 'Tarih Bilinmiyor')
            entry_id = s.get('entry_id', '')
            
            if surname and surname in full_report:
                full_report += f"- {first_author} et al. ({published[:4]}). **{s['title']}**. arXiv: {entry_id}\n"
                seen_refs.add(s['title'])
    
    if not seen_refs:
        for s in state.get("arxiv_summaries", []):
            if s['entry_id'] in selected_ids and s['title'] not in seen_refs:
                authors = s.get('authors', ['Bilinmeyen Yazar'])
                first_author = authors[0] if authors else 'Bilinmeyen Yazar'
                published = s.get('published', 'Tarih Bilinmiyor')
                entry_id = s.get('entry_id', '')
                full_report += f"- {first_author} et al. ({published[:4]}). **{s['title']}**. arXiv: {entry_id}\n"
                seen_refs.add(s['title'])
    
    if not seen_refs:
        full_report += "- *Akademik kaynak bulunamadı.*\n"
    
    # 3b. Web Kaynakları — sadece raporda atıf yapılanlar
    full_report += "\n### Web Kaynakları\n"
    web_sources = state.get("web_sources", [])
    seen_web = set()
    
    for src in web_sources:
        url = src.get("url", "")
        title = src.get("title", "Başlıksız")
        if url and url not in seen_web:
            title_words = [w for w in title.split() if len(w) >= 3 and w[0].isupper()]
            is_cited = any(word in full_report for word in title_words) if title_words else True
            
            if is_cited:
                full_report += f"- {title}. Erişim: {url}\n"
                seen_web.add(url)
    
    if not seen_web:
        for src in web_sources:
            url = src.get("url", "")
            title = src.get("title", "Başlıksız")
            if url and url not in seen_web:
                full_report += f"- {title}. Erişim: {url}\n"
                seen_web.add(url)
    
    if not seen_web:
        full_report += "- *Web kaynağı bulunamadı.*\n"

    print(f"✅ Rapor Hazır. (Akademik: {len(seen_refs)}, Web: {len(seen_web)} kaynak)")
    
    return {
        "final_report": full_report,
        "is_complete": True
    }