import os
import requests
import tempfile
import fitz  
import time
from llama_parse import LlamaParse

# LlamaParse bulut tabanlı çalıştığı için VRAM/GPU yönetimi gereksiz.
# Paralel worker sayısı — LlamaParse kendi bulut sunucularında çalıştığı için
# yerel kaynak kısıtlaması yok, 4 worker güvenli bir değer.
LLAMA_PARSE_WORKERS = 4

def download_pdf(pdf_url: str) -> str:
    """
    Verilen URL'den PDF dosyasını indirir ve geçici bir dosyaya kaydeder.
    Bağlantı hatalarına karşı 3 deneme yapar.
    """
    print(f"İndiriliyor: {pdf_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            fd, path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, 'wb') as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
            return path
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ İndirme Hatası ({pdf_url}): {e}. {wait_time} sn sonra tekrar deneniyor... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ İndirme Başarısız: {e}")
                return None
    return None

def fallback_pdf_to_markdown(file_path: str) -> str:
    """
    LlamaParse başarısız olursa veya API Key yoksa kullanılacak yedek yöntem (PyMuPDF).
    Tabloları ve görselleri mükemmel alamaz ama metni kurtarır.
    """
    print(f"⚠️ PyMuPDF (Yedek) devreye giriyor...")
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        
        return f"# PDF Content (Extracted via PyMuPDF)\n\n{text}"
    except Exception as e:
        return f"HATA: Yedek yöntem de başarısız oldu. {e}"

def parse_pdf_to_markdown(file_path: str) -> str:
    """
    Bulutta çalışan LlamaParse ile PDF'i Markdown'a çevirir.
    Başarısız olursa PyMuPDF'e düşer.
    Gerekli env: LLAMA_CLOUD_API_KEY
    """
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    
    if not api_key:
        print("⚠️ LLAMA_CLOUD_API_KEY bulunamadı! Yedek yönteme (PyMuPDF) geçiliyor.")
        return fallback_pdf_to_markdown(file_path)

    print(f"☁️ LlamaParse (Bulut) ile işleniyor: {file_path}")
    
    try:
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            num_workers=LLAMA_PARSE_WORKERS,
            verbose=False,
            language="en"  # Akademik makaleler İngilizce
        )
        
        documents = parser.load_data(file_path)
        
        if not documents:
            print("⚠️ LlamaParse boş sonuç döndürdü, yedek yönteme geçiliyor.")
            return fallback_pdf_to_markdown(file_path)
        
        # Tüm sayfaları birleştir
        markdown_text = "\n\n".join([doc.text for doc in documents])
        
        print(f"✅ LlamaParse başarılı ({len(documents)} sayfa işlendi).")
        return markdown_text
    
    except Exception as e:
        print(f"❌ LlamaParse Hatası: {e}")
        return fallback_pdf_to_markdown(file_path)

def process_paper(pdf_url: str) -> str:
    """Makaleyi indir, parse et ve içeriği döndür."""
    local_path = download_pdf(pdf_url)
    if not local_path:
        return "Dosya indirilemedi."
    
    content = parse_pdf_to_markdown(local_path)
    
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except:
            pass
        
    return content