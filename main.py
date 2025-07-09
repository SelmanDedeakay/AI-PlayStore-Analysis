import streamlit as st
import pandas as pd
from scraper import scrape_reviews
from analyzer import sentiment_analysis, is_fake_review, is_interesting_review

st.set_page_config(layout="wide")
st.title("Joygame Review Intelligence")

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
    "Top Race : Car Battle Racing": "com.flatgames.toprace",
    "Cross'em All": "com.flatgames.crossemall",
    "Dog Whisperer: Fun Walker Game": "com.flatgames.dogwhisperer"
}

with st.sidebar:
    st.header("Select a Game")
    game_choice = st.selectbox(
        "Choose a game for review analysis:",
        options=list(games.keys()),
        index=0
    )
    app_id = games[game_choice]

tab_manual, tab_full = st.tabs(["Manual Review Test", "Full Review Analysis"])

with tab_manual:
    st.header(f"Manual Review Test - {game_choice}")
    review_input = st.text_area("Enter a review to analyze:", height=120)

    if st.button("Analyze Review"):
        if not review_input.strip():
            st.warning("Please enter some review text!")
        else:
            with st.spinner("Analyzing review..."):
                sentiment = sentiment_analysis(review_input)
                fake, fake_reason = is_fake_review(review_input)
                interesting, interesting_reason = is_interesting_review(review_input)

            st.subheader("Results")
            st.markdown(f"- **Sentiment:** {sentiment['label']} (score: {sentiment['score']:.2f})")
            st.markdown(f"- **Fake Review:** {fake} ({fake_reason})")
            st.markdown(f"- **Interesting Review:** {interesting} ({interesting_reason})")

with tab_full:
    st.header(f"Full Review Analysis - {game_choice}")

    if st.button("Scrape and Analyze All Reviews"):
        with st.spinner("Scraping reviews... this may take a while"):
            try:
                reviews = scrape_reviews(app_id)
                if not reviews:
                    st.warning("No reviews found!")
                else:
                    st.info(f"Scraped {len(reviews)} reviews. Starting analysis...")
                    data = []
                    for r in reviews:
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
                    df = pd.DataFrame(data)
                    st.success("Analysis complete!")

                    st.dataframe(df.head(20))

                    csv = df.to_csv(index=False).encode('utf-8')
                    json_data = df.to_json(orient='records')

                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{app_id}_reviews_analysis.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{app_id}_reviews_analysis.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Error during scraping/analysis: {e}")
