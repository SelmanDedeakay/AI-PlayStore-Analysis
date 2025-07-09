# scraper.py

from google_play_scraper import reviews_all
import pandas as pd
from tqdm import tqdm

APP_ID = "com.flatgames.patrolofficer"

def scrape_reviews(app_id=APP_ID, langs=None):
    if langs is None:
        # Google Play için yaygın dillerden örnek liste
        langs = [
    "en", "tr", "es", "de", "fr", "ru", "pt", "ar", "zh", "hi", "ja", "ko", "id",
    "it", "vi", "th", "pl", "ro", "uk", "ms", "fa", "nl", "cs", "hu", "he", "sv"
]


    all_reviews = []

    print(f"Scraping reviews for app: {app_id}")
    for lang in tqdm(langs, desc="Languages"):
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
