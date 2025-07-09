import streamlit as st
from analyzer import sentiment_analysis, is_fake_review, is_interesting_review

st.set_page_config(layout="wide")
st.title("Joygame Review Intelligence Demo")

# Oyun bilgileri
games = {
    "Patrol Officer - Cop Simulator": {
        "app_id": "com.flatgames.patrolofficer",
        "stars": 3.7
    },
    "Desert Warrior": {
        "app_id": "com.flatgames.desertwarrior",
        "stars": 4.6
    },
    "Arcade Ball.io - Let's Bowl!": {
        "app_id": "com.flatgames.arcadeballio",
        "stars": 3.7
    },
    "Wrestling Trivia Run": {
        "app_id": "com.flatgames.wrestlingtrivia",
        "stars": 4.1
    },
    "Chips Factory - Tycoon Game": {
        "app_id": "com.flatgames.chipsfactory",
        "stars": 3.2
    },
    "Deck Dash: Epic Card Battle RP": {
        "app_id": "com.flatgames.deckdash",
        "stars": 4.5
    },
    "Wedding Rush 3D!": {
        "app_id": "com.flatgames.weddingrush3d",
        "stars": 4.2
    },
    "Hospital Life": {
        "app_id": "com.flatgames.hospitallife",
        "stars": 5.0
    },
    "1001 Brain Zen Puzzles": {
        "app_id": "com.flatgames.brainzenpuzzles",
        "stars": 3.6
    },
    "Wand Evolution: Magic Mage Run": {
        "app_id": "com.flatgames.wandevolution",
        "stars": 3.8
    },
    "Take'em Down!": {
        "app_id": "com.flatgames.takeemdown",
        "stars": None
    },
    "Top Race : Car Battle Racing": {
        "app_id": "com.flatgames.toprace",
        "stars": 4.2
    },
    "Cross'em All": {
        "app_id": "com.flatgames.crossemall",
        "stars": 3.1
    },
    "Dog Whisperer: Fun Walker Game": {
        "app_id": "com.flatgames.dogwhisperer",
        "stars": 4.0
    }
}

with st.sidebar:
    st.header("Select a Game")
    game_choice = st.selectbox(
        "Choose a game for review analysis:",
        options=list(games.keys()),
        index=list(games.keys()).index("Patrol Officer - Cop Simulator")
    )

    game_data = games[game_choice]
    st.markdown(f"**{game_choice}**")
    if game_data["stars"] is not None:
        st.markdown(f"⭐ {game_data['stars']}")

    st.markdown("---")

    st.header("Project FAQ")

    faq = {
        "Which NLP models or libraries did you use for review analysis?":
        "- Sentiment analysis için nlptown/bert-base-multilingual-uncased-sentiment modeli kullandık.\n"
        "- Fake ve ilginç yorum tespiti için Google Gemini 2.5 Flash LLM API’sini kullandık.",

        "What strategy did you follow for fake review detection?":
        "- Kısmi rule-based duplicate kontrolü ile birlikte LLM tabanlı dilsel değerlendirme.\n"
        "- Embedding ve cosine similarity ile yakın duplicate tespiti yapılıyor.",

        "How did you compute sentiment scores?":
        "- Mevcut nlptown BERT tabanlı model kullanıldı, çıktı 1-5 yıldız üzerinden label’a dönüştürüldü.",

        "What logic or methods did you use to select interesting reviews?":
        "- LLM’e esnek prompt ile mizah, yaratıcılık, öneri gibi içeriklerin tespiti istendi.\n"
        "- Manuel keyword ve uzunluk bazlı basit kurallar da var (fallback olarak).",

        "How did you implement review scraping? Is the data continuously updatable?":
        "- Python google-play-scraper kütüphanesiyle tüm dillerde yorumlar çekiliyor.\n"
        "- Bu işlem batch şeklinde yapılıyor, periyodik otomasyonla güncellenebilir.",

        "If you had to make this scalable and real-time, how would you architect the system?":
        "- Streaming API + Pub/Sub tabanlı mimari kurulabilir.\n"
        "- Yorumlar anlık çekilip, Kafka veya benzeri kuyruğa konur.\n"
        "- Analizler microservice şeklinde ölçeklenir ve sonuçlar gerçek zamanlı dashboard’a aktarılır."
    }

    for q, a in faq.items():
        st.markdown(f"**{q}**")
        st.markdown(a)
        st.markdown("---")

# Ana alan - Demo
st.header(f"Test a Review - {game_choice}")

review_input = st.text_area("Enter a review to analyze:", height=120)

if st.button("Analyze"):
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
