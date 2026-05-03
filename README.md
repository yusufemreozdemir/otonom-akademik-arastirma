# Otonom Akademik Araştırma Ajanı

## Projenin Amacı

Bu proje, akademik araştırma yapmak isteyen öğrencilere, araştırmacılara ve akademisyenlere güçlü bir **başlangıç noktası** sunmak amacıyla geliştirilmiştir. Manuel olarak yapıldığında saatler hatta günler sürebilen literatür tarama, makale okuma ve bilgi sentezleme süreçlerini otonom bir şekilde dakikalar seviyesine indirgemeyi hedefler. Büyük dil modellerini çok katmanlı bir yapay zeka ajanı (LangGraph) mimarisiyle birleştirerek, saatler sürebilen araştırmaları kapsamlı ve yüksek standartlı raporlara dönüştürür.

## Teknik Çözümler

Sistem, geleneksel RAG (Retrieval-Augmented Generation) veya doğrudan LLM kullanımında karşılaşılan birçok darboğazı çözmek üzere tasarlanmıştır. LLM'lerin multimodal yapay zeka yetenekleri sayesinde grafikler, matematiksel formüller ve sayfa düzeni hiçbir veri kaybına uğramadan doğrudan bağlama dahil edilir.

*   **Modüler Yazım Algoritması:** LLM'lerin sınırlı çıktı (output token) kapasitesini aşmak için *Analyst Node* üç ayrı modda çalışır: Mimar (içindekileri belirler), Yazar (bölümleri parça parça yazar) ve Editör (parçaları birleştirir). Bu sayede model bağlamdan kopmaz ve halüsinasyon görmeden uzun ve tutarlı raporlar üretebilir. Modüler yazım algoritması, her bir bölümün (giriş, metodoloji, bulgular, tartışma) ayrı ayrı üretilip ardından birleştirilmesini sağlayarak LLM'lerin kısıtlı output token limitlerini aşmadan derinlemesine analiz yapabilmesine olanak tanır.
*   **Dinamik API Yönetimi:** Ücretsiz API kullanımlarında sıkça karşılaşılan limit sorunlarına karşı özel bir `APIKeyManager` sınıfı geliştirilmiştir. Beklenmedik kotaların dolması durumunda sistem çökmez; anahtarları otonom bir şekilde rotasyona sokarak veya bir süre bekleyerek araştırmasına devam eder. Bu yapı, kullanıcıların birden fazla ücretsiz API anahtarını tek bir havuzda yönetmesine olanak tanıyarak kesintisiz araştırma imkanı sunar.
*   **Akademik Kalite Kontrol:** Yazılan raporların doğruluğu, bir *Reviewer Node* (Kalite Kontrol Uzmanı) tarafından denetlenir. Kaynakça hataları veya yüzeysel analizler tespit edilirse rapor onaylanmaz ve düzeltilmesi için geri gönderilir. Onaylanan raporlar, Pandoc ve Typst motorlarıyla bilimsel dizgi formatında (.pdf) çıktılanır.

## Nasıl İndirilir ve Kullanılır?

### 1. Gereksinimler
*   **Python 3.10** veya üzeri bir sürüm. (tercihen 3.12)
*   PDF dönüştürme özelliği için sisteminizde **Pandoc** ve **Typst**'ın kurulu olması tavsiye edilir.

### 2. Kurulum
Projeyi bilgisayarınıza klonlayın ve dizine gidin:
```bash
git clone https://github.com/yusufemreozdemir/otonom-akademik-arastirma.git
cd otonom-akademik-arastirma
```

Sanal ortam (virtual environment) oluşturup bağımlılıkları yükleyin. (Bu işlem için `uv` kullanmanız tavsiye edilir):
```bash
# uv kullanıyorsanız:
uv venv
uv pip install -r requirements.txt

# Standart pip kullanıyorsanız:
python -m venv venv
# Windows için:
venv\Scripts\activate
# MacOS/Linux için:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Yapılandırma (.env)
Proje ana dizininde bir `.env` dosyası oluşturun ve API anahtarlarınızı tanımlayın. Kota limitlerine takılmadan uzun süreli araştırmalar yapabilmek için birden fazla API anahtarı ile çalışmanız önerilir. API anahtarlarınızı örnekteki şekilde numaralandırarak ekleyebilirsiniz.

```env
GOOGLE_API_KEY_1=sizin_gemini_anahtariniz_1
GOOGLE_API_KEY_2=sizin_gemini_anahtariniz_2
TAVILY_API_KEY=sizin_tavily_anahtariniz
```

### 4. Çalıştırma
Her şey hazır olduğunda sistemi başlatın:
```bash
python main.py
```

Sistem çalıştıktan sonra size hangi konuyu araştırmak istediğinizi soracaktır. Konuyu girdikten sonra ajanın literatür taraması yapmasını, PDF'leri indirip analiz etmesini ve raporu yazmasını terminal üzerinden adım adım takip edebilirsiniz.

Oluşturulan sonuçlar, proje dizinindeki `reports/` klasörünün altına Markdown (`.md`) ve PDF formatında otomatik olarak kaydedilecektir.

## Sınırlandırmalar ve Katkı
Proje şu anda öncelikli olarak İngilizce makale veritabanlarını (ArXiv vb.) kullanarak en iyi sonucu vermektedir. Sistemdeki arama araçlarını, okuma yöntemlerini veya dil modellerini kendi araştırma tarzınıza göre kolayca modifiye edebilir, projenin geliştirilmesine katkıda bulunabilirsiniz.
