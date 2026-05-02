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
    queries = protocol.get("search_queries", []) 
    topic = state.get("final_topic", state.get("user_topic", "arastirma"))
    
    # Eğer sorgu yoksa kullanıcının girdiği sorguyu kullan
    if not queries:
        queries = [state["user_topic"]]

    # Araştırma sonuçlarının tutulacağı değişkenler
    all_arxiv_data = []
    web_context = ""
    all_web_sources = [] # Web kaynaklarını ayrıca topla (referans bölümü için)
    arxiv_consecutive_fails = 0

    # Sorgular için araştırma döngüsü
    for q in queries:
        # ArXiv'den her sorgu için veri çek (Eğer çok fazla hata almadıysak)
        if arxiv_consecutive_fails < 2:
            results = search_arxiv(query = q, max_results = 25) 
            if results:
                all_arxiv_data.extend(results)
                arxiv_consecutive_fails = 0 # Başarılı olursa sayacı sıfırla
            else:
                arxiv_consecutive_fails += 1
                print(f"⚠️ Uyarı: '{q}' sorgusu için ArXiv'den sonuç alınamadı.")
        else:
            print(f"⏭️ ArXiv servis dışı veya limitli olduğu için '{q}' için ArXiv atlanıyor.")

        # Web araması yap (ArXiv başarısız olsa bile devam et)
        web_text, web_sources = search_web(query = q)
        web_context += f"\n--- Search Query: {q} ---\n{web_text}"
        all_web_sources.extend(web_sources)

        # Bir sonraki sorgu öncesi bekleme (Rate limit koruması)
        # Eğer çok hata aldıysak uzun bekle, yoksa normal bekle
        sleep_time = random.randint(20, 30) if arxiv_consecutive_fails >= 2 else random.randint(10, 20)
        time.sleep(sleep_time)

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