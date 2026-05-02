import os
from typing import List, Dict
from tavily import TavilyClient
import arxiv

tavily_client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY")) # Tavily Client Başlatma


def search_arxiv(query: str, max_results: int = 100) -> List[Dict]:
    """
    ArXiv üzerinde akademik makale arar.
    Metadataları (Başlık, Özet, Yazar, PDF Linki) döndürür.
    429 (Too Many Requests) hatalarına karşı dirençlidir.
    """
    import time
    from arxiv import Client, Search, SortCriterion

    print(f"ArXiv aranıyor: {query}")
    
    # Daha muhafazakar bir client yapılandırması
    client = Client(
        page_size=min(max_results, 50), # Sayfa boyutunu küçülterek yükü azaltalım
        delay_seconds=5.0, # ArXiv API'sine fazla yüklenmemek için bekleme süresi artırıldı
        num_retries=3
    )
    
    search = Search(
        query = query,
        max_results = max_results,
        sort_by = SortCriterion.Relevance
    )

    max_manual_retries = 3
    for attempt in range(max_manual_retries):
        try:
            results = []
            # Client.results() generator olduğu için döngüye girince istek atar
            for result in client.results(search):
                results.append({
                    "title": result.title,
                    "summary": result.summary.replace("\n", " "),
                    "published": result.published.strftime("%Y-%m-%d"),
                    "authors": [author.name for author in result.authors],
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id
                })
            return results
            
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str:
                wait_time = (attempt + 1) * 30 # Bekleme süresi artırıldı (30, 60, 90 saniye)
                print(f"⚠️ ArXiv Rate Limit (429) yakalandı. {wait_time} saniye bekleniyor... (Deneme {attempt+1}/{max_manual_retries})")
                time.sleep(wait_time)
            elif "connection" in error_str or "10054" in error_str:
                print(f"⚠️ ArXiv Bağlantı Hatası: {e}. Tekrar deneniyor...")
                time.sleep(5)
            else:
                print(f"❌ ArXiv Aramasında Beklenmedik Hata: {e}")
                if attempt == max_manual_retries - 1:
                    return []
                time.sleep(5)
    
    return []

def search_web(query: str) -> tuple:
    """
    Tavily kullanarak web araması yapar.
    Returns: (context_text: str, sources: list[dict]) — hem metin hem kaynak listesi döner.
    """
    import time
    import os
    from tavily import TavilyClient

    print(f"Web aranıyor (Tavily): {query}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            
            response = client.search(
                query = query,
                search_depth = "advanced",
                max_results = 5,
                include_answer = True
            )
            
            context = f"Tavily AI Summary: {response.get('answer', '')}\n\nSources:\n"
            sources = []
            for result in response.get("results", []):
                context += f"- Title: {result['title']}\n  Content: {result['content']}\n  URL: {result['url']}\n\n"
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                })
            return context, sources

        except Exception as e:
            error_str = str(e).lower()
            if "10054" in error_str or "connection" in error_str or "reset" in error_str:
                print(f"⚠️ Tavily Bağlantı Hatası (TCP Reset): {e}. Yeniden bağlanılıyor... (Deneme {attempt+1}/{max_retries})")
                time.sleep(5)
            elif "429" in error_str:
                print(f"⚠️ Tavily Rate Limit. Bekleniyor...")
                time.sleep(10)
            else:
                print(f"❌ Tavily Hatası: {e}")
                if attempt == max_retries - 1:
                    return "Web araması başarısız oldu.", []
                time.sleep(2)
                
    return "Web araması zaman aşımına uğradı veya bağlantı sağlanamadı.", []