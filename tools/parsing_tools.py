import os
import requests
import tempfile
import fitz  
import gc
import torch
import threading
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from markdownify import markdownify as md

# Thread-safe VRAM temizliği için global lock
_vram_lock = threading.Lock()

def get_pdf_thread_count() -> int:
    """
    GPU VRAM boyutuna göre optimal PDF thread sayısını hesaplar.
    Kural: Her 2 GB VRAM için 1 thread (min: 1, max: 4).
    GPU yoksa CPU mode: 2 thread.
    """
    try:
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            thread_count = max(1, min(4, int(vram_gb // 2)))
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🖥️ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) → {thread_count} paralel PDF thread")
            return thread_count
        else:
            print("🖥️ GPU bulunamadı → CPU mode, 2 paralel PDF thread")
            return 2
    except Exception as e:
        print(f"⚠️ Thread sayısı hesaplanamadı ({e}), varsayılan: 2")
        return 2

# Modül yüklenince bir kez hesapla — tüm threadler bu değeri paylaşır
PDF_THREAD_COUNT = get_pdf_thread_count()

def download_pdf(pdf_url: str) -> str:
    """
    Verilen URL'den PDF dosyasını indirir ve geçici bir dosyaya kaydeder.
    Bağlantı hatalarına karşı 3 deneme yapar.
    """
    import time
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
            error_str = str(e).lower()
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
    Docling çalışmazsa kullanılacak Hızlı ve Güvenli yöntem (PyMuPDF).
    Tabloları mükemmel alamaz ama metni kurtarır.
    """
    print(f"Docling başarısız oldu, PyMuPDF (Yedek) devreye giriyor...")
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        
        # Metni biraz temizleyip Markdown'a benzetelim
        return f"# PDF Content (Extracted via PyMuPDF)\n\n{text}"
    except Exception as e:
        return f"HATA: Yedek yöntem de başarısız oldu. {e}"

def parse_pdf_to_markdown(file_path: str) -> str:
    """
    Önce Docling dener, olmazsa PyMuPDF kullanır.
    """
    print(f"PDF işleniyor: {file_path}")
    
    # Ana yöntem: Docling 
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        
        # GPU ve OCR yapılandırması
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device="cuda") # GPU kullanımı (cuda/auto)
        pipeline_options.do_ocr = True # OCR'ı aktif et
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        # --- THREAD-SAFE BELLEK TEMİZLİĞİ ---
        with _vram_lock:
            del result
            del converter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
        # -------------------------------------
        
        print("Docling başarıyla (GPU desteğiyle) dönüştürdü. VRAM temizlendi.")
        return markdown_text
    
    # Yedek yöntem: PyMuPDF
    except Exception as e:
        print(f"Docling Hatası: {e}")
        return fallback_pdf_to_markdown(file_path) # Hata ne olursa olsun Yedek Yönteme geç

def process_paper(pdf_url: str) -> str:
    local_path = download_pdf(pdf_url)
    if not local_path:
        return "Dosya indirilemedi."
    
    content = parse_pdf_to_markdown(local_path)
    
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except:
            pass # Silinemezse de program durmasın
        
    return content