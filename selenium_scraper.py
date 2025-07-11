from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
import time
from tqdm import tqdm

APP_ID = "com.flatgames.patrolofficer"

default_langs = [
    "en", "tr", "es", "de", "fr", "ru", "pt", "ar", "zh", "hi", "ja", "ko", "id", "it", "vi", "th", "pl", "ro",
    "uk", "ms", "fa", "nl", "cs", "hu", "he", "sv", "fi", "no", "da", "el", "sr", "hr", "bg", "sk", "lt", "lv",
    "sl", "et", "bn", "ta", "te", "ml", "mr", "gu", "kn", "pa", "am", "sw", "zu", "xh", "af", "az", "ka", "ur",
    "km", "lo", "my", "ne", "si", "is", "ga", "sq", "mk", "hy", "mn", "ps", "kk", "uz", "tg"
]

def init_driver(lang="en"):
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--lang={lang}")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # options.add_argument("--headless=new")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def check_page_loaded_successfully(driver, lang):
    """Sayfanın başarıyla yüklenip yüklenmediğini kontrol et"""
    try:
        # Sayfa title'ını kontrol et
        title = driver.title.lower()
        
        # Error sayfası kontrolü
        error_indicators = [
            "not found", "404", "error", "unavailable", 
            "bulunamadı", "hata", "nicht gefunden", "non trouvé",
            "no encontrado", "не найдено"
        ]
        
        if any(indicator in title for indicator in error_indicators):
            print(f"   [!] {lang}: Error sayfası tespit edildi - {driver.title}")
            return False
        
        # Play Store sayfası olup olmadığını kontrol et
        if "play.google.com" not in driver.current_url:
            print(f"   [!] {lang}: Play Store sayfası değil")
            return False
        
        # Ana içerik yüklenmiş mi kontrol et
        try:
            WebDriverWait(driver, 10).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-g-id]")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "c-wiz")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#ow8"))
                )
            )
            print(f"   ✓ {lang}: Sayfa başarıyla yüklendi")
            return True
            
        except TimeoutException:
            print(f"   [!] {lang}: Sayfa içeriği yüklenemedi (timeout)")
            return False
            
    except Exception as e:
        print(f"   [!] {lang}: Sayfa kontrolü hatası - {e}")
        return False

def check_app_available_in_language(driver, lang):
    """Uygulamanın o dilde mevcut olup olmadığını kontrol et"""
    try:
        # Uygulama bulunamadı mesajları
        not_found_selectors = [
            "//div[contains(text(), 'not found')]",
            "//div[contains(text(), 'bulunamadı')]", 
            "//div[contains(text(), 'nicht gefunden')]",
            "//div[contains(text(), 'non trouvé')]",
            "//div[contains(text(), 'no encontrado')]",
            "//div[contains(text(), 'не найдено')]",
            ".error-section",
            "[data-error-code]"
        ]
        
        for selector in not_found_selectors:
            try:
                if selector.startswith("//"):
                    elements = driver.find_elements(By.XPATH, selector)
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    print(f"   [!] {lang}: Uygulama bu dilde mevcut değil")
                    return False
            except:
                continue
        
        # Uygulama adının varlığını kontrol et
        try:
            app_title = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1, [data-g-id] h1, .Fd93Bb"))
            )
            if app_title.text.strip():
                print(f"   ✓ {lang}: Uygulama mevcut")
                return True
        except TimeoutException:
            print(f"   [!] {lang}: Uygulama başlığı bulunamadı")
            return False
            
    except Exception as e:
        print(f"   [!] {lang}: Uygulama kontrolü hatası - {e}")
        return False
    
    return True

def scroll_until_devices_found(driver, max_scrolls=10):
    """Cihaz butonları bulunana kadar scroll et"""
    device_selectors = [
        'div[role="button"][id^="formFactor_"]',
        '.D3Qfie[role="button"]',
        '[aria-label="Phone"], [aria-label="Tablet"], [aria-label="Chromebook"]'
    ]
    
    for scroll_attempt in range(max_scrolls):
        for selector in device_selectors:
            try:
                device_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                if device_buttons:
                    return device_buttons
            except:
                continue
        
        current_position = driver.execute_script("return window.pageYOffset")
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(1)
        
        new_position = driver.execute_script("return window.pageYOffset")
        if current_position == new_position:
            break
    
    return []

def find_see_all_reviews_button(driver, timeout=5):
    """Tüm yorumları gör butonunu bul"""
    
    # Spesifik selector - share butonuna tıklamayacak
    review_button_selectors = [
        # Ana spesifik selector
        '#yDmH0d > c-wiz.SSPGKf.Czez9d > div > div > div:nth-child(1) > div > div.wkMJlb.YWi3ub > div > div.qZmL0 > div:nth-child(1) > c-wiz:nth-child(5) > section > div > div.Jwxk6d > div:nth-child(5) > div > div > button',
        
        # Biraz daha kısa alternatif
        'c-wiz:nth-child(5) > section > div > div.Jwxk6d > div:nth-child(5) > div > div > button',
        
        # Section içindeki Jwxk6d altındaki 5. div'deki buton
        'section > div > div.Jwxk6d > div:nth-child(5) > div > div > button',
        
        # Fallback - eğer yapı değişirse
        '.Jwxk6d > div:nth-child(5) > div > div > button'
    ]
    
    for selector in review_button_selectors:
        try:
            button = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            print(f"   ✓ Yorumlar butonu bulundu: '{button.text.strip()}'")
            return button
            
        except TimeoutException:
            continue
        except Exception:
            continue
    
    print(f"   [!] Yorumlar butonu bulunamadı")
    return None

def scroll_and_extract_reviews(driver, lang, device, max_scrolls=50):
    """Modal içinde scroll yap ve yorumları çıkar"""
    reviews = []
    
    # Modal scroll container'ları
    scroll_selectors = [
        "/html/body/div[5]/div[2]/div/div/div/div/div[2]",
        "div[role='dialog'] div[style*='overflow']",
        ".VfPpkd-wzTsW div[style*='overflow-y']"
    ]
    
    scroll_container = None
    for selector in scroll_selectors:
        try:
            if selector.startswith("/"):
                scroll_container = driver.find_element(By.XPATH, selector)
            else:
                scroll_container = driver.find_element(By.CSS_SELECTOR, selector)
            break
        except:
            continue
    
    if not scroll_container:
        print(f"   [!] Scroll container bulunamadı")
        return reviews
    
    print(f"   📜 Scroll başlıyor...")
    
    last_height = 0
    unchanged_count = 0
    
    for scroll_num in range(max_scrolls):
        # Scroll et
        current_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", scroll_container)
        time.sleep(1.5)
        
        # Yeni height'ı kontrol et
        new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)
        
        if new_height == last_height:
            unchanged_count += 1
            if unchanged_count >= 3:
                print(f"   🛑 Scroll tamamlandı ({scroll_num + 1} scroll)")
                break
        else:
            unchanged_count = 0
        
        last_height = new_height
        
        # Progress
        if scroll_num % 10 == 0:
            current_reviews = driver.find_elements(By.CLASS_NAME, "h3YV2d")
            print(f"   Scroll {scroll_num + 1}: {len(current_reviews)} yorum")
    
    # Yorumları çıkar - HİÇBİR FİLTRE YOK
    try:
        review_elements = driver.find_elements(By.CLASS_NAME, "h3YV2d")
        print(f"   ✅ {len(review_elements)} yorum elementi bulundu")
        
        for idx, element in enumerate(review_elements):
            try:
                content = element.text.strip()
                # Hiçbir filtreleme yapmıyoruz, hepsini alıyoruz
                reviews.append({
                    "lang": lang,
                    "device": device,
                    "content": content,  # Boş olsa bile al
                    "length": len(content),
                    "element_index": idx  # Hangi sırada olduğunu da kaydedelim
                })
            except Exception as e:
                # Hata olsa bile boş bir entry ekle
                reviews.append({
                    "lang": lang,
                    "device": device,
                    "content": f"[ERROR: {str(e)}]",
                    "length": 0,
                    "element_index": idx
                })
                continue
        
        print(f"   ✅ {len(reviews)} yorum çekildi (filtresiz)")
                
    except Exception as e:
        print(f"   [!] Yorum çıkarma hatası: {e}")
    
    return reviews

def scrape_language(lang):
    """Belirli bir dil için tüm cihazlardan yorumları çek"""
    all_reviews = []
    driver = None
    
    try:
        driver = init_driver(lang)
        url = f"https://play.google.com/store/apps/details?id={APP_ID}&hl={lang}&gl=US"
        print(f"\n🌐 {lang.upper()} - Sayfa açılıyor...")
        
        # Sayfayı yükle
        try:
            driver.get(url)
            time.sleep(3)
        except WebDriverException as e:
            print(f"   [X] {lang}: Sayfa yüklenemedi - {e}")
            return all_reviews
        
        # Sayfa başarıyla yüklendi mi kontrol et
        if not check_page_loaded_successfully(driver, lang):
            return all_reviews
        
        # Uygulama bu dilde mevcut mu kontrol et
        if not check_app_available_in_language(driver, lang):
            return all_reviews
        
        # Cihaz butonlarını bul
        device_buttons = scroll_until_devices_found(driver)
        
        if not device_buttons:
            print(f"   [!] {lang}: Cihaz butonları bulunamadı")
            return all_reviews
        
        print(f"   📱 {len(device_buttons)} cihaz bulundu")
        
        # Her cihaz için yorumları çek
        for idx, button in enumerate(device_buttons):
            try:
                # Cihaz bilgilerini al
                aria_label = button.get_attribute("aria-label") or f"device_{idx+1}"
                print(f"\n   🔄 Cihaz: {aria_label}")
                
                # Cihaza tıkla
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", button)
                time.sleep(2)
                
                # Yorumlar butonunu ara
                review_button = find_see_all_reviews_button(driver)
                
                if not review_button:
                    print(f"   [!] Yorumlar butonu bulunamadı")
                    continue
                
                # Yorumlar modal'ını aç
                driver.execute_script("arguments[0].click();", review_button)
                print(f"   ✓ Yorumlar modal açıldı")
                time.sleep(3)
                
                # Scroll ve yorumları çek
                reviews = scroll_and_extract_reviews(driver, lang, aria_label)
                print(f"   ✅ {len(reviews)} yorum çekildi")
                all_reviews.extend(reviews)
                
                # Modal'ı kapat
                try:
                    driver.find_element(By.TAG_NAME, 'body').send_keys('\ue00c')  # ESC
                    time.sleep(2)
                except:
                    pass
                
            except Exception as e:
                print(f"   [!] Cihaz {idx+1} hatası: {e}")
                continue
        
    except Exception as e:
        print(f"   [X] {lang}: Genel hata - {e}")
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
    
    return all_reviews

def main():
    """Ana scraping fonksiyonu"""
    all_reviews = []
    successful_langs = 0
    failed_langs = []
    
    print(f"🚀 Google Play Store Multi-Language Scraper")
    print(f"📱 Uygulama: {APP_ID}")
    print(f"🌍 {len(default_langs)} dil işlenecek")
    
    for lang in tqdm(default_langs, desc="🌍 Diller"):
        try:
            reviews = scrape_language(lang)
            if reviews:
                all_reviews.extend(reviews)
                successful_langs += 1
                print(f"   ✅ {lang}: {len(reviews)} yorum eklendi")
            else:
                print(f"   ❌ {lang}: Yorum bulunamadı")
                failed_langs.append(lang)
                
        except KeyboardInterrupt:
            print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
            break
        except Exception as e:
            print(f"   [X] {lang}: Beklenmeyen hata - {e}")
            failed_langs.append(lang)
            continue
    
    # Sonuçları kaydet
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        
        # Duplikatları temizle
        original_count = len(df)
        df = df.drop_duplicates(subset=['content'])
        
        filename = f"google_play_reviews_{APP_ID}_{len(df)}_reviews.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        
        print(f"\n🎉 Scraping tamamlandı!")
        print(f"📊 {len(df)} benzersiz yorum ({original_count - len(df)} duplikat temizlendi)")
        print(f"🌍 {successful_langs}/{len(default_langs)} dil başarılı")
        print(f"💾 Dosya: {filename}")
        
        # Başarısız dilleri göster
        if failed_langs:
            print(f"\n❌ Başarısız diller ({len(failed_langs)}):")
            print(f"   {', '.join(failed_langs)}")
        
        # İstatistikler
        if len(df) > 0:
            print(f"\n📈 En çok yorum olan diller:")
            lang_counts = df.groupby('lang').size().sort_values(ascending=False)
            for lang, count in lang_counts.head(10).items():
                print(f"   {lang}: {count} yorum")
                
            print(f"\n📱 Cihaz bazında:")
            device_counts = df.groupby('device').size().sort_values(ascending=False)
            for device, count in device_counts.items():
                print(f"   {device}: {count} yorum")
                
            # Ortalama yorum uzunluğu
            avg_length = df['length'].mean()
            print(f"\n📏 Ortalama yorum uzunluğu: {avg_length:.1f} karakter")
            
    else:
        print("\n❌ Hiç yorum çekilemedi!")
        print("Olası nedenler:")
        print("- Uygulama hiçbir dilde mevcut değil")
        print("- Ağ bağlantısı problemi")
        print("- Google Play Store yapısı değişmiş olabilir")
        print("- Rate limiting")

if __name__ == "__main__":
    main()