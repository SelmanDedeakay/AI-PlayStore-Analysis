import streamlit as st
import pandas as pd
from scraper import scrape_reviews
from analyzer import sentiment_analysis, is_fake_review, is_interesting_review

st.set_page_config(layout="wide")
st.title("🎮 Joygame Review Intelligence")

# Game list
games = {
    "Patrol Officer - Cop Simulator": "com.flatgames.patrolofficer",
    "Desert Warrior": "com.flatgames.desertwarrior",
    "Arcade Ball.io - Let's Bowl!": "com.flatgames.arcadeballio",
    "Wrestling Trivia Run": "com.flatgames.wrestlingtrivia",
    "Chips Factory - Tycoon Game": "com.flatgames.chipsfactory",
    "Deck Dash: Epic Card Battle RP": "com.flatgames.deckdash",
    "Wedding Rush 3D!": "com.flatgames.weddingrush3d",
    "Hospital Life": "com.flatgames.hospitallife",
    "1001 Brain Zen Puzzles": "com.flatgames.brainzenpuzzles",
    "Wand Evolution: Magic Mage Run": "com.flatgames.wandevolution",
    "Take'em Down!": "com.flatgames.takeemdown",
    "Top Race : Car Battle Racing": "com.flatgames.toprace",
    "Cross'em All": "com.flatgames.crossemall",
    "Dog Whisperer: Fun Walker Game": "com.flatgames.dogwhisperer"
}

# Sidebar: game selection
with st.sidebar:
    st.header("🎯 Game Selection")
    game_choice = st.selectbox("Choose a game:", list(games.keys()), index=0)
    app_id = games[game_choice]

    st.markdown("---")
    with st.expander("📘 Questions & Answers", expanded=False):
        st.markdown("""
**• Which NLP models or libraries did you use for review analysis?**  
We used a combination of BERT for sentiment classification and Gemini LLM for fake & interesting review detection.

**• What strategy did you follow for fake review detection?**  
Duplicate detection (exact + TF-IDF similarity) + Gemini LLM-based reasoning.

**• How did you compute sentiment scores?**  
We used a multilingual BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) returning star-based scores.

**• What logic or methods did you use to select interesting reviews?**  
LLM-based scoring based on humor, creativity, novelty, constructive feedback, and tone.

**• How did you implement review scraping?**  
We use the `google-play-scraper` library to extract all reviews in many supported languages.

**• If you had to make this scalable and real-time, how would you architect the system?**  
We'd use background workers (Celery), a vector DB for deduplication (e.g., Qdrant), async APIs (FastAPI), and caching/load-balancing layers.
        """)

# Tabs
tab_manual, tab_full = st.tabs(["📝 Manual Review", "📊 Full Review Analysis"])

# ---- MANUAL ANALYSIS ----
with tab_manual:
    st.header(f"📝 Manual Review – {game_choice}")
    review_input = st.text_area("Enter a review to analyze:", height=120)

    if st.button("🔍 Analyze Review"):
        if not review_input.strip():
            st.warning("Please enter review text.")
        else:
            with st.spinner("Analyzing..."):
                sentiment = sentiment_analysis(review_input)
                fake, fake_reason = is_fake_review(review_input)
                interesting, interesting_reason = is_interesting_review(review_input)

            st.subheader("🧠 Result")
            st.markdown(f"- **Sentiment:** `{sentiment['label']}` (score: `{sentiment['score']:.2f}`)")
            st.markdown(f"- **Fake Review:** `{fake}` ({fake_reason})")
            st.markdown(f"- **Interesting:** `{interesting}` ({interesting_reason})")

# ---- FULL REVIEW ANALYSIS ----
with tab_full:
    st.header(f"📊 Full Review Analysis – {game_choice}")

    if st.button("🚀 Scrape & Analyze All Reviews"):
        with st.spinner("Scraping reviews..."):
            try:
                reviews = scrape_reviews(app_id)
                if not reviews:
                    st.warning("No reviews found.")
                else:
                    st.info(f"{len(reviews)} reviews fetched. Running analysis...")
                    data = []
                    total = len(reviews)
                    progress_bar = st.progress(0)

                    for i, r in enumerate(reviews):
                        text = r.get("content", "")
                        if not text.strip():
                            continue
                        sent = sentiment_analysis(text)
                        fake, fake_reason = is_fake_review(text)
                        interesting, interesting_reason = is_interesting_review(text)

                        data.append({
                            "review": text,
                            "sentiment_label": sent["label"],
                            "sentiment_score": sent["score"],
                            "is_fake": fake,
                            "fake_reason": fake_reason,
                            "is_interesting": interesting,
                            "interesting_reason": interesting_reason
                        })

                        progress_bar.progress((i + 1) / total)

                    progress_bar.empty()
                    df = pd.DataFrame(data)
                    st.success("✅ Analysis complete!")
                    st.dataframe(df.head(20), use_container_width=True)

                    # Downloads
                    csv = df.to_csv(index=False).encode('utf-8')
                    json_data = df.to_json(orient='records')

                    st.download_button("⬇️ Download CSV", csv, file_name="reviews_analysis.csv", mime="text/csv")
                    st.download_button("⬇️ Download JSON", json_data, file_name="reviews_analysis.json", mime="application/json")

            except Exception as e:
                st.error(f"❌ Error during scraping/analysis: {e}")
