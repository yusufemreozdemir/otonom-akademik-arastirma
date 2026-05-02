import json
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

# YENİLİK: Tek string yerine Bölüm Nesnesi
class SectionOutput(BaseModel):
    title: str = Field(description="Bölümün başlığı.")
    content: str = Field(description="Bölümün markdown içeriği (En az 400 kelime).")

# YENİLİK: Liste yapısı. Model bu listeyi doldurmak zorunda hissedecek.
class BatchContent(BaseModel):
    sections: List[SectionOutput] = Field(description="Yazılan bölümlerin listesi.")

# --- ANA DÜĞÜM ---

def analyst_node(state: ResearchState):
    print("--- ANALYST NODE ÇALIŞIYOR (1x1 Mode) ---")
    
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    
    # 1. ADIM: MİMAR (Outline Yoksa)
    if not outline:
        return run_architect_mode(state)
    
    # 2. ADIM: YAZAR (Daha yazılacak bölüm varsa)
    elif current_index < len(outline):
        return run_batch_writer(state)
        
    # 3. ADIM: EDİTÖR (Tüm bölümler bitti)
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

def _escape_braces(text):
    """LangChain template'lerinde kullanılacak f-string değerlerindeki
    süslü parantezleri escape eder ({X} -> {{X}}) böylece
    ChatPromptTemplate bunları değişken olarak yorumlamaz."""
    if isinstance(text, str):
        return text.replace("{", "{{").replace("}", "}}")
    elif isinstance(text, list):
        return [_escape_braces(item) for item in text]
    return str(text).replace("{", "{{").replace("}", "}}")

def run_batch_writer(state: ResearchState):
    outline = state.get("report_outline", [])
    current_index = state.get("current_section_index", 0)
    
    # Tek bölüm yaz — kalite ve kontrol için en güvenli yöntem
    target_title = outline[current_index]
    topic = state.get("final_topic", state.get("user_topic"))
    
    print(f"✍️ Mod: YAZAR (Bölüm {current_index+1}/{len(outline)}: '{target_title[:40]}...')")

    llm = get_model()
    structured_llm = llm.with_structured_output(SectionOutput)
    
    full_contents_text = str(state.get("full_paper_contents", {}))[:100000]
    web_text = state.get("web_data", "")[:30000]
    
    # Süslü parantez içeren başlıkları escape et
    safe_title = _escape_braces(target_title)
    safe_topic = _escape_braces(topic)
    
    # Daha önce yazılmış bölümlerin başlıklarını bağlam olarak ver
    written_so_far = list(state.get("written_sections", {}).keys())
    prev_context = ""
    if written_so_far:
        prev_context = f"Şimdiye kadar yazılan bölümler: {_escape_braces(str(written_so_far))}. Tekrar yazmaya gerek yok."
    
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
    10. YASAK ATIFLAR: 'Analiz', 'Analizler', 'Araştırma', 'Kaynak', 'Değerlendirme', 'İnceleme', 'Tavily AI Summary'
        kelimeleri ASLA atıf olarak kullanılamaz. Bunlar kaynak değildir.
    {prev_context}
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
    
    # Outline başlığıyla kaydet (LLM'in değiştirdiği başlığı KULLANMA)
    new_written = state.get("written_sections", {}).copy()
    new_written[target_title] = response.content
    print(f"   -> Bölüm yazıldı: {target_title[:40]}...")
    
    return {
        "written_sections": new_written,
        "current_section_index": current_index + 1,  # Sabit artış — her zaman 1
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
    for title in outline:
        content = written_sections.get(title)
        
        # Eğer birebir başlık yoksa, benzerini ara
        if not content:
            for k, v in written_sections.items():
                if k in title or title in k:
                    content = v
                    break
        
        if content:
            full_report += f"## {title}\n\n{content}\n\n"
        else:
            full_report += f"## {title}\n\n*(Bu bölüm oluşturulurken teknik bir aksaklık yaşandı)*\n\n"

    # 2. REFERANSLAR BÖLÜMÜ (Akademik + Web kaynakları)
    full_report += "## REFERANSLAR\n\n"
    
    # 2a. Akademik Kaynaklar (ArXiv)
    full_report += "### Akademik Kaynaklar\n"
    seen_refs = set()
    selected_ids = state.get("selected_paper_ids", [])
    
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
    
    # 2b. Web Kaynakları (Tavily)
    full_report += "\n### Web Kaynakları\n"
    web_sources = state.get("web_sources", [])
    seen_web = set()
    
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
        "is_complete": True # Graph Reviewer'a geçecek
    }