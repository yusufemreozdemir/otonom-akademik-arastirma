from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os
import glob
import pypandoc
from state import ResearchState
from model import get_model

class ReviewOutput(BaseModel):
    is_satisfactory: bool = Field(description = "Rapor, belirlenen standartlara ve plana uygun mu?")
    feedback: str = Field(description = "Eğer eksik varsa detaylı eleştiri, yoksa onay mesajı.")
    score: int = Field(description = "Rapora 100 üzerinden verilen akademik kalite notu.")
    failed_sections: List[str] = Field(
        description = "Sorunlu bölümlerin BAŞLIKLARI listesi. Sadece yeniden yazılması gereken bölümleri belirt. Sorun yoksa boş liste.",
        default_factory=list
    )

def reviewer_node(state: ResearchState):
    print("--- REVIEWER NODE: KALİTE KONTROL UZMANI ---")
    
    llm = get_model()
    structured_llm = llm.with_structured_output(ReviewOutput)
    
    # TÜM BAĞLAMI ÇEKİYORUZ
    user_topic = state.get("user_topic", "Bilinmiyor")
    final_topic = state.get("final_topic", "Bilinmiyor")
    protocol = state.get("research_protocol", {})
    report = state.get("final_report", "")
    
    # Bağlamı stringe çevirelim
    protocol_str = str(protocol.get("search_queries", []))
    
    system_prompt = """Sen bir akademik rapor kalite kontrol uzmanısın.
        Görevin, yazılan araştırma raporunu belirli kalite kriterlerine göre değerlendirmek.

        DEĞERLENDİRME KRİTERLERİ:

        1. KAYNAK BÜTÜNLÜĞÜ (En Kritik Kriter): 
           - Raporda metin içinde atıf yapılan HER kaynak REFERANSLAR bölümünde OLMALIDIR. (Kaynakçada olmayan uydurma atıflar YASAKTIR).
           - ANCAK, Referanslar listesindeki her kaynağın metinde geçmesi ZORUNLU DEĞİLDİR (Bunlar 'İncelenen Eserler' kabul edilir). Bu yüzden "Kullanılmayan Kaynaklar" var diyerek SAKIN PUAN KIRMA.
           - Metin içinde kaynakçada olmayan uydurma yazarlara (halüsinasyon) atıf yapılmışsa puan kır ve o bölümü yeniden yazılması için işaretle.
        
        2. KONU UYUMU: Rapor, araştırma konusuna odaklı mı? Konu dağılmış mı yoksa tutarlı mı?
        
        3. AKADEMİK KALİTE: 
           - Akademik dil ve teknik derinlik yeterli mi?
           - Yüzeysel veya tekrarlayan ifadeler var mı?
           - Bölümler TAM MI? Cümle veya paragraf ortasında kesilmiş bölüm var mı?
        
        4. FORMAT VE YAPI: 
           - Bölüm başlıkları mantıklı bir sırayla mı? LaTeX doğru kullanılmış mı?
           - Rapor bütünlüklü ve okunabilir mi?
           - Eksik veya yarım kalmış bölüm var mı?

        PUANLAMA:
        - 80-100: Rapor yayınlanabilir kalitede.
        - 60-79: Küçük düzeltmelerle kabul edilebilir.
        - 40-59: Önemli eksiklikler var, revizyon gerekli.
        - 0-39: Ciddi sorunlar var, yeniden yazılmalı.
        
        KARAR:
        - is_satisfactory = True: Puan 60 veya üzeriyse.
        - is_satisfactory = False: Puan 60'ın altındaysa. Hangi bölümde ne sorun olduğunu detaylıca açıkla.
        
        SORUNLU BÖLÜMLER:
        - failed_sections listesine SADECE yeniden yazılması gereken bölümlerin BAŞLIKLARINI ekle.
        - Başlıkları rapordaki ## işaretinden sonraki haliyle AYNEN yaz.
        - ÖNEMLİ: Eğer bir bölümde yanlış atıf, halüsinasyon veya eksik kaynak sorunu varsa O BÖLÜMÜ de listeye ekle.
        - Sadece İÇERİK EKSİKLİĞİ (yarım kalma) DEĞİL, KAYNAKÇA HATALARI olan bölümleri de failed_sections'a dahil et.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Kullanıcı İsteği: {user_topic}\nNetleşen Konu: {final_topic}\n\nRAPOR:\n{report}\n\nDEĞERLENDİR:")
    ])
    
    # Context sınırını koruyarak veriyoruz
    chain = prompt | structured_llm
    response = chain.invoke({
        "user_topic": user_topic,
        "final_topic": final_topic,
        "report": report
    })
    
    print(f"\n📊 KALİTE RAPORU")
    print(f"   Puan: {response.score}/100")
    print(f"   Karar: {'✅ GEÇTİ' if response.is_satisfactory else '❌ KALDI'}")
    if response.failed_sections:
        print(f"   Sorunlu Bölümler: {response.failed_sections}")
    print(f"   Feedback: {response.feedback}\n")

    if response.is_satisfactory:
        try:
            report_title = state.get("report_title", state.get("final_topic", "arastirma_raporu"))
            
            report_dir = "reports"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            safe_title = "".join([c if c.isalnum() else "_" for c in report_title])[:60]
            filename_md = os.path.join(report_dir, f"{safe_title}.md")
            filename_pdf = os.path.join(report_dir, f"{safe_title}.pdf")
            
            # 1. Save Markdown
            with open(filename_md, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"📄 Markdown Rapor kaydedildi: {os.path.abspath(filename_md)}")
            
            # 2. Setup Pandoc and Typst Paths
            winget_pkg_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
            typst_paths = glob.glob(os.path.join(winget_pkg_dir, "Typst.Typst*", "**", "typst.exe"), recursive=True)
            pandoc_paths = glob.glob(os.path.join(winget_pkg_dir, "JohnMacFarlane.Pandoc*", "**", "pandoc.exe"), recursive=True)
            
            if typst_paths and pandoc_paths:
                typst_dir = os.path.dirname(typst_paths[0])
                pandoc_exe = pandoc_paths[0]
                
                # Add typst directory to PATH so pandoc can find it
                if typst_dir not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + typst_dir
                    
                os.environ["PYPANDOC_PANDOC"] = pandoc_exe
                
                # 3. Convert to PDF using Typst
                print("⚙️ PDF oluşturuluyor (Pandoc + Typst)...")
                pypandoc.convert_text(
                    report,
                    'pdf',
                    format='md',
                    outputfile=filename_pdf,
                    extra_args=['--pdf-engine=typst']
                )
                print(f"📄 PDF Rapor oluşturuldu: {os.path.abspath(filename_pdf)}")
            else:
                print("⚠️ Uyarı: Sistemde Pandoc veya Typst bulunamadığı için PDF oluşturulamadı.")
                
        except Exception as e:
            print(f"❌ PDF dönüştürme hatası: {e}")

    return {
        "feedback": response.feedback,
        "failed_section_titles": response.failed_sections,
        "is_complete": response.is_satisfactory 
    }