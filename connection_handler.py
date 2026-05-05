"""
İnternet bağlantı kesintisi yönetim modülü.
Sistem çalışırken internet kesilirse kullanıcıya bilgi verir,
tekrar denemek isteyip istemediğini sorar ve bağlantı gelince
kaldığı yerden devam eder.
"""

import socket
import time
import urllib.request

# ANSI renk kodları
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def check_internet(timeout=5) -> bool:
    """
    İnternet bağlantısının olup olmadığını kontrol eder.
    Google DNS (8.8.8.8) ve bir HTTP isteği ile çift katmanlı kontrol yapar.
    """
    # Katman 1: DNS/Socket kontrolü (hızlı)
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        pass
    
    # Katman 2: HTTP kontrolü (yedek)
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        pass
    
    return False


def wait_for_connection(context_message: str = "") -> bool:
    """
    İnternet bağlantısı kesildiğinde kullanıcıya interaktif bir bekleme döngüsü sunar.
    
    - Kullanıcıya 'Tekrar dene (y/n)' sorusu sorar.
    - 'y' derse bağlantıyı kontrol eder; bağlantı yoksa uyarı verip tekrar sorar.
    - 'n' derse False döndürür (işlem iptal).
    - Bağlantı gelirse True döndürür (kaldığı yerden devam).
    
    Args:
        context_message: Hangi işlem sırasında kesildiğini belirten mesaj.
    
    Returns:
        True: Bağlantı geldi, devam edilebilir.
        False: Kullanıcı iptal etti.
    """
    print(f"\n{'='*60}")
    print(f"{Colors.RED}{Colors.BOLD}⚠️  İNTERNET BAĞLANTISI KESİLDİ!{Colors.RESET}")
    if context_message:
        print(f"{Colors.YELLOW}   📍 Kesinti anı: {context_message}{Colors.RESET}")
    print(f"{Colors.CYAN}   💾 Mevcut ilerleme kaydedildi. Bağlantı gelince kaldığı yerden devam edilecek.{Colors.RESET}")
    print(f"{'='*60}")
    
    while True:
        try:
            answer = input(f"\n{Colors.YELLOW}🔄 Tekrar dene (y/n): {Colors.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.RED}❌ İşlem kullanıcı tarafından iptal edildi.{Colors.RESET}")
            return False
        
        if answer == 'n':
            print(f"{Colors.RED}❌ İşlem kullanıcı tarafından iptal edildi.{Colors.RESET}")
            return False
        elif answer == 'y':
            print(f"{Colors.CYAN}🔍 Bağlantı kontrol ediliyor...{Colors.RESET}", end=" ", flush=True)
            
            if check_internet():
                print(f"{Colors.GREEN}{Colors.BOLD}✅ Bağlantı sağlandı! Kaldığı yerden devam ediliyor...{Colors.RESET}")
                return True
            else:
                print(f"{Colors.RED}❌ Bağlantı hâlâ yok!{Colors.RESET}")
                print(f"{Colors.YELLOW}   ℹ️  İnternet bağlantınızı kontrol edin ve tekrar deneyin.{Colors.RESET}")
                # Döngü tekrar başa döner, kullanıcıya tekrar sorar
        else:
            print(f"{Colors.YELLOW}   ℹ️  Lütfen 'y' (evet) veya 'n' (hayır) girin.{Colors.RESET}")


def is_connection_error(exception: Exception) -> bool:
    """
    Verilen exception'ın internet bağlantı hatası olup olmadığını tespit eder.
    """
    error_msg = str(exception).lower()
    
    connection_keywords = [
        "connectionerror",
        "connection refused",
        "connection reset",
        "connection aborted",
        "connectionreseterror",
        "networkerror",
        "network is unreachable",
        "name or service not known",
        "nodename nor servname",
        "temporary failure in name resolution",
        "getaddrinfo failed",
        "no route to host",
        "errno 11001",       # Windows DNS çözümleme hatası
        "errno 11004",       # Windows DNS hatası
        "10053",             # Windows: Bağlantı kesildi
        "10054",             # Windows: Uzak taraf bağlantıyı kapattı
        "10060",             # Windows: Bağlantı zaman aşımı
        "10061",             # Windows: Bağlantı reddedildi
        "urlopen error",
        "remote end closed connection",
        "remotedisconnected",
        "newconnectionerror",
        "maxretryerror",
        "sslerror",          # SSL hatası bazen bağlantı kopmasından olur
        "readtimeout",
        "connecttimeout",
        "sockettimeout",
        "socket.gaierror",
        "oserror",
        "brokenpipeerror",
        "incompleteread",
        "[errno -2]",        # DNS çözümleme hatası (Linux)
        "[errno -3]",        # DNS çözümleme hatası (Linux)
    ]
    
    # Exception tipi kontrolü
    connection_exceptions = (
        ConnectionError,
        ConnectionResetError,
        ConnectionAbortedError,
        ConnectionRefusedError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
        OSError,
    )
    
    if isinstance(exception, connection_exceptions):
        return True
    
    # Mesaj içeriği kontrolü
    for keyword in connection_keywords:
        if keyword in error_msg:
            return True
    
    return False


def handle_connection_error(exception: Exception, context: str = "") -> bool:
    """
    Bağlantı hatası yakalandığında çağrılır.
    Eğer gerçekten bir internet kesintisi ise kullanıcıya sorar.
    
    Args:
        exception: Yakalanan hata.
        context: Hatanın oluştuğu bağlam açıklaması.
    
    Returns:
        True: Tekrar denenebilir.
        False: Kullanıcı iptal etti veya bağlantı hatası değil.
    """
    if not is_connection_error(exception):
        return False
    
    # Gerçekten internet yok mu doğrula (geçici bir hıçkırık olabilir)
    if check_internet():
        # İnternet var ama hata aldık — geçici bir sunucu sorunu olabilir
        return False
    
    # İnternet gerçekten kesilmiş
    return wait_for_connection(context)
