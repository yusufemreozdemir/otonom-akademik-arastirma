from state import ResearchState
from tools.search_tools import search_arxiv, search_web
import time, random, os
from datetime import datetime

def researcher_node(state: ResearchState):
    """
    Bu düğümde LLM yok, sadece Python kodu çalışır.
    Manager'ın belirlediği protokolü uygular.
    """
    print("--- RESEARCHER NODE ÇALIŞIYOR ---")
    
    # Manager'ın belirlediği arama terimlerini al
    protocol = state.get("research_protocol", {})
    arxiv_queries = protocol.get("arxiv_queries") or []
    tavily_queries = protocol.get("tavily_queries") or []
    
    # Geriye dönük uyumluluk veya LLM hatası durumunda fallback
    if not arxiv_queries and not tavily_queries:
        old_queries = protocol.get("search_queries") or [state.get("user_topic", "research")]
        arxiv_queries = old_queries
        tavily_queries = old_queries

    topic = state.get("final_topic", state.get("user_topic", "arastirma"))
    
    # Araştırma sonuçlarının tutulacağı değişkenler
    all_arxiv_data = []
    web_context = ""
    all_web_sources = [] 
    arxiv_consecutive_fails = 0

    # 1. ArXiv Aramaları
    print(f"--- ArXiv Aramaları Başlatılıyor ({len(arxiv_queries)} sorgu) ---")
    for q in arxiv_queries:
        if arxiv_consecutive_fails < 2:
            results = search_arxiv(query = q, max_results = 25) 
            if results:
                all_arxiv_data.extend(results)
                arxiv_consecutive_fails = 0
            else:
                arxiv_consecutive_fails += 1
                print(f"⚠️ Uyarı: '{q}' sorgusu için ArXiv'den sonuç alınamadı.")
        else:
            print(f"⏭️ ArXiv servis dışı veya limitli olduğu için '{q}' atlanıyor.")
        
        # Rate limit koruması
        time.sleep(random.randint(10, 20))

    # 2. Tavily Web Aramaları
    print(f"--- Tavily Web Aramaları Başlatılıyor ({len(tavily_queries)} sorgu) ---")
    for q in tavily_queries:
        web_text, web_sources = search_web(query = q)
        web_context += f"\n--- Search Query: {q} ---\n{web_text}"
        all_web_sources.extend(web_sources)
        
        # Rate limit koruması
        time.sleep(random.randint(5, 10))

    # Tavily çıktılarını kaydet
    save_tavily_data(web_context, topic)
    
    # Web kaynaklarını tekilleştir (aynı URL'den gelen tekrarları kaldır)
    unique_web_sources = []
    seen_urls = set()
    for src in all_web_sources:
        if src["url"] not in seen_urls:
            unique_web_sources.append(src)
            seen_urls.add(src["url"])

    # Sonuçları state'e ekle
    if not all_arxiv_data:
        print("❌ KRİTİK UYARI: Hiçbir ArXiv makalesi toplanamadı! Rapor kalitesi düşük olabilir.")
    else:
        print(f"--- Toplanan Makale Sayısı: {len(all_arxiv_data)} ---")
    
    print(f"--- Toplanan Web Kaynağı: {len(unique_web_sources)} ---")
    
    return {
        "arxiv_summaries": all_arxiv_data, # Ham liste
        "web_data": web_context, # Birleştirilmiş metin veri
        "web_sources": unique_web_sources # Yapılandırılmış kaynak listesi [{title, url}]
    }

def save_tavily_data(content, topic):
    """Tavily çıktılarını sonradan incelemek için dosyaya kaydeder."""
    output_dir = "tavily_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c if c.isalnum() else "_" for c in topic])[:30]
    filename = f"tavily_{safe_topic}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Tavily Search Results for: {topic}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(content)
        print(f"🌐 Tavily çıktıları kaydedildi: {filepath}")
    except Exception as e:
        print(f"⚠️ Tavily çıktıları kaydedilirken hata oluştu: {e}")