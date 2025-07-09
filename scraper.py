# scraper.py

from google_play_scraper import reviews_all
import pandas as pd
from tqdm import tqdm

APP_ID = "com.flatgames.patrolofficer"
default_langs = [
    "en",  # English
    "tr",  # Turkish
    "es",  # Spanish
    "de",  # German
    "fr",  # French
    "ru",  # Russian
    "pt",  # Portuguese
    "ar",  # Arabic
    "zh",  # Chinese
    "hi",  # Hindi
    "ja",  # Japanese
    "ko",  # Korean
    "id",  # Indonesian
    "it",  # Italian
    "vi",  # Vietnamese
    "th",  # Thai
    "pl",  # Polish
    "ro",  # Romanian
    "uk",  # Ukrainian
    "ms",  # Malay
    "fa",  # Persian
    "nl",  # Dutch
    "cs",  # Czech
    "hu",  # Hungarian
    "he",  # Hebrew
    "sv",  # Swedish
    "fi",  # Finnish
    "no",  # Norwegian
    "da",  # Danish
    "el",  # Greek
    "sr",  # Serbian
    "hr",  # Croatian
    "bg",  # Bulgarian
    "sk",  # Slovak
    "lt",  # Lithuanian
    "lv",  # Latvian
    "sl",  # Slovenian
    "et",  # Estonian
    "bn",  # Bengali
    "ta",  # Tamil
    "te",  # Telugu
    "ml",  # Malayalam
    "mr",  # Marathi
    "gu",  # Gujarati
    "kn",  # Kannada
    "pa",  # Punjabi
    "am",  # Amharic
    "sw",  # Swahili
    "zu",  # Zulu
    "xh",  # Xhosa
    "af",  # Afrikaans
    "az",  # Azerbaijani
    "ka",  # Georgian
    "ur",  # Urdu
    "km",  # Khmer
    "lo",  # Lao
    "my",  # Burmese
    "ne",  # Nepali
    "si",  # Sinhala
    "is",  # Icelandic
    "ga",  # Irish
    "sq",  # Albanian
    "mk",  # Macedonian
    "hy",  # Armenian
    "mn",  # Mongolian
    "ps",  # Pashto
    "kk",  # Kazakh
    "uz",  # Uzbek
    "tg",  # Tajik
]

def scrape_reviews(app_id=APP_ID, langs=None):
    if langs is None:
        # Google Play için yaygın dillerden örnek liste
        langs = default_langs


    all_reviews = []

    print(f"Scraping reviews for app: {app_id}")
    for lang in tqdm(default_langs, desc="Languages"):
        try:
            reviews = reviews_all(
                app_id,
                lang=lang,
                country='us',  # Lokasyon burada dil uyumu için önemli
                sleep_milliseconds=0,
            )
            for r in reviews:
                all_reviews.append({
                    "userName": r.get("userName", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                    "at": r.get("at"),
                    "lang": lang
                })
        except Exception as e:
            print(f"[!] Error scraping lang={lang}: {e}")

    return all_reviews


if __name__ == "__main__":
    reviews = scrape_reviews()
    df = pd.DataFrame(reviews)
    df.to_csv("reviews_raw.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ {len(df)} reviews saved to reviews_raw.csv")
