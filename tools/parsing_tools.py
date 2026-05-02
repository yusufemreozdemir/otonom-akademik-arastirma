import os
import requests
import time

# PDF'lerin kaydedileceği kalıcı dizin
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")

def ensure_papers_dir():
    os.makedirs(PAPERS_DIR, exist_ok=True)

def download_pdf(pdf_url: str, paper_id: str = None) -> str:
    """
    PDF'i indirir ve kalıcı dosya yolunu döndürür.
    paper_id verilirse dosya adı olarak kullanır (tekrar indirmeyi önler).
    """
    ensure_papers_dir()
    
    if paper_id:
        safe_name = paper_id.replace("/", "_").replace(":", "_").replace("?", "_")
        file_path = os.path.join(PAPERS_DIR, f"{safe_name}.pdf")
    else:
        import tempfile
        fd, file_path = tempfile.mkstemp(suffix=".pdf", dir=PAPERS_DIR)
        os.close(fd)
    
    # Zaten indirilmişse tekrar indirme
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"📄 Zaten indirilmiş, atlanıyor: {os.path.basename(file_path)}")
        return file_path
    
    print(f"İndiriliyor: {pdf_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return file_path
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ İndirme Hatası ({pdf_url}): {e}. {wait_time} sn sonra tekrar deneniyor... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ İndirme Başarısız: {e}")
                return None
    return None

def process_paper(pdf_url: str, paper_id: str = None) -> str:
    """Makaleyi indir ve dosya yolunu döndür. Artık markdown'a çevirme yok."""
    return download_pdf(pdf_url, paper_id=paper_id)