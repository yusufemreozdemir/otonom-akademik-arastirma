import os
import markdown
from xhtml2pdf import pisa

def save_as_pdf(markdown_text: str, output_filename: str):
    """
    Markdown metnini alır, akademik CSS ile HTML'e çevirir ve PDF olarak kaydeder.
    """
    print(f"PDF Dönüştürme Başlatılıyor: {output_filename}")
    
    # 1. Markdown'ı HTML'e Çevir (Tablo ve uzantı desteğiyle)
    html_content = markdown.markdown(
        markdown_text, 
        extensions = ['tables', 'fenced_code', 'toc']
    )

    # 2. Akademik Stil (CSS) Tanımla
    # Türkçe karakterler için Arial/Helvetica kullanıyoruz, Windows'ta genelde sorunsuzdur.
    css_style = """
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
            @top-center {
                content: "Otonom Akademik Araştırma Raporu";
                font-family: Helvetica, Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
            @bottom-center {
                content: "Sayfa " counter(page);
                font-family: Helvetica, Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
        }
        
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            text-align: justify;
        }

        h1 {
            color: #2c3e50;
            font-size: 24pt;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
            margin-top: 0;
            text-align: center;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 16pt;
            margin-top: 25px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #34495e;
            font-size: 13pt;
            margin-top: 20px;
        }
        
        p {
            margin-bottom: 15px;
        }
        
        ul, ol {
            margin-bottom: 15px;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border: 1px solid #eee;
            border-radius: 5px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            color: #333;
            font-weight: bold;
        }
        
        .meta-info {
            text-align: center;
            font-size: 10pt;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
    </style>
    """

    # 3. HTML Şablonunu Birleştir
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        {css_style}
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """

    # 4. PDF Olarak Kaydet
    try:
        with open(output_filename, "wb") as result_file:
            # Encoding hatası almamak için src encoding belirtiyoruz
            pisa_status = pisa.CreatePDF(
                src = full_html, 
                dest = result_file,
                encoding = 'utf-8' 
            )

        if pisa_status.err:
            print("PDF oluşturulurken hata oluştu.")
            return False
        
        print(f"PDF Başarıyla Oluşturuldu: {output_filename}")
        return True
        
    except Exception as e:
        print(f"PDF Yazma Hatası: {e}")
        return False