import gradio as gr
import pandas as pd
# Make sure you have a scraper.py file with a scrape_reviews function
from scraper import scrape_reviews 
# Updated to use the correct function names from your latest analyzer.py
from analyzer import hybrid_fake_review_detection, detect_interesting_review, analyze_sentiment
import os

# NOTE: The game list remains the same.
games = {
    "Patrol Officer - Cop Simulator": "com.flatgames.patrolofficer",
    "Desert Warrior": "com.desertwarrior.joygame",
    "Arcade Ball.io - Let's Bowl!": "com.oskankayirci.skeeball",
    "Wrestling Trivia Run": "com.wrestling.trivia.run",
    "Chips Factory - Tycoon Game": "com.maveragames.ChipsFactory",
    "Deck Dash: Epic Card Battle RP": "com.arvisgames.deckdash",
    "Wedding Rush 3D!": "com.oskankayirci.weddingrush",
    "Hospital Life": "com.oskankayirci.hospitallife",
    "1001 Brain Zen Puzzles": "com.mafiagames.brainzenpuzzle",
    "Wand Evolution: Magic Mage Run": "com.easyclapgames.wandevolution",
    "Take'em Down!": "com.flatgames.takeemdown",
    "Top Race : Car Battle Racing": "com.Funrika.TopRace",
    "Cross'em All": "com.flatgames.crossemall",
    "Dog Whisperer: Fun Walker Game": "com.oskankayirci.petcollector"
}

def analyze_single_review(review_text):
    """Analyzes a single review for fake, interesting status, and sentiment."""
    if not review_text.strip():
        return "Please enter review text.", "", ""
    
    try:
        fake_result = hybrid_fake_review_detection(review_text)
        interesting_result = detect_interesting_review(review_text)
        sentiment_result = analyze_sentiment(review_text)
        
        # Simplified output formatting to match the actual analyzer output
        fake_output = f"""**🔍 Fake Review Detection:**
- **Status:** {'🚨 SUSPICIOUS/FAKE' if fake_result['is_fake'] else '✅ GENUINE'}
- **Type:** {fake_result['label']}
- **Confidence:** {fake_result['confidence']:.3f}
- **Reason:** {fake_result['reason']}"""
        
        interesting_output = f"""**⭐ Interesting Content Detection:**
- **Status:** {'🌟 INTERESTING' if interesting_result['is_interesting'] else '📄 STANDARD'}
- **Type:** {interesting_result['label']}
- **Analysis:** {interesting_result['reason']}"""
        
        # Add sentiment analysis output
        sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        sentiment_output = f"""**😊 Sentiment Analysis:**
- **Sentiment:** {sentiment_emoji.get(sentiment_result['sentiment'], '😐')} {sentiment_result['sentiment']}"""
        
        # Add detailed sentiment if available
        if 'detailed_sentiment' in sentiment_result:
            sentiment_output += f"\n- **Detailed:** {sentiment_result['detailed_sentiment']}"
        
        sentiment_output += f"""
- **Polarity Score:** {sentiment_result['polarity']} (Range: -1.0 to +1.0)
- **Confidence:** {sentiment_result['confidence']:.3f}
- **Method:** {sentiment_result['method']}"""
        
        # Add raw scores if available (from multilingual model)
        if 'raw_scores' in sentiment_result:
            sentiment_output += f"\n- **Raw Scores:** {sentiment_result['raw_scores']}"
        
        return fake_output, interesting_output, sentiment_output
    except Exception as e:
        return f"Error: {str(e)}", "", ""

def analyze_all_reviews(game_choice, progress=gr.Progress()):
    """Scrapes and analyzes all reviews for a chosen game."""
    if not game_choice:
        return "Please select a game.", None, None
    
    app_id = games[game_choice]
    
    try:
        progress(0, desc="Scraping reviews...")
        reviews = scrape_reviews(app_id)
        
        if not reviews:
            return "No reviews found.", None, None
        
        progress(0.1, desc=f"Found {len(reviews)} reviews. Starting analysis...")
        
        data = []
        #total = min(len(reviews), 50)  # Limit to 50 reviews for demo
        
        for i, r in enumerate(reviews):
            text = r.get("content", "")
            if not text.strip():
                continue
            
            progress((i + 1) / total, desc=f"Processing review {i + 1}/{total}")
            
            fake_result = hybrid_fake_review_detection(text)
            interesting_result = detect_interesting_review(text)
            sentiment_result = analyze_sentiment(text)
            
            # DataFrame structure now includes sentiment analysis
            data.append({
                "review": text[:200] + "..." if len(text) > 200 else text,
                "is_fake": fake_result["is_fake"],
                "fake_label": fake_result["label"],
                "fake_confidence": round(fake_result["confidence"], 3),
                "is_interesting": interesting_result["is_interesting"],
                "interesting_label": interesting_result["label"],
                "reason_for_classification": interesting_result['reason'],
                "sentiment": sentiment_result["sentiment"],
                "detailed_sentiment": sentiment_result.get("detailed_sentiment", sentiment_result["sentiment"]),
                "polarity_score": sentiment_result["polarity"],
                "sentiment_confidence": round(sentiment_result["confidence"], 3),
                "sentiment_method": sentiment_result["method"]
            })
        
        df = pd.DataFrame(data)
        
        # The summary dataframe now shows only the available columns
        summary_df = df.copy()
        
        progress(1.0, desc="Analysis complete!")
        
        return f"✅ Analysis complete! Processed {len(data)} reviews.", summary_df, df
        
    except Exception as e:
        return f"❌ Error during scraping/analysis: {str(e)}", None, None

# Create the Gradio interface
with gr.Blocks(title="🎮 Joygame Review Intelligence", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎮 Joygame Review Intelligence")
    
    with gr.Tabs():
        with gr.TabItem("📝 Manual Review"):
            gr.Markdown("## 📝 Manual Review Analysis")
            
            with gr.Row():
                with gr.Column(scale=2):
                    review_input = gr.Textbox(
                        label="Enter a review to analyze:",
                        placeholder="e.g., 'This game is amazing! I love it so much.'",
                        lines=5
                    )
                    analyze_btn = gr.Button("🔍 Analyze Review", variant="primary")
                
                with gr.Column(scale=3):
                    # Updated to three outputs including sentiment
                    fake_output = gr.Markdown(label="Fake Review Detection")
                    interesting_output = gr.Markdown(label="Interesting Review Detection")
                    sentiment_output = gr.Markdown(label="Sentiment Analysis")
            
            analyze_btn.click(
                fn=analyze_single_review,
                inputs=[review_input],
                outputs=[fake_output, interesting_output, sentiment_output]
            )
        
        with gr.TabItem("📊 Full Review Analysis"):
            gr.Markdown("## 📊 Full Review Analysis")
            
            with gr.Row():
                game_dropdown = gr.Dropdown(
                    choices=list(games.keys()),
                    label="🎯 Choose a game:",
                    value=list(games.keys())[0]
                )
                scrape_btn = gr.Button("🚀 Scrape & Analyze All Reviews", variant="primary")
            
            status_output = gr.Textbox(label="Status", interactive=False)
            
            # The summary table is now the main output, with the detailed one in an accordion
            summary_table = gr.Dataframe(
                label="📋 Analysis Results (Top 50 Reviews)",
                wrap=True
            )
            
            # Using an accordion for the full downloadable data
            with gr.Accordion("🧾 Download Full Results", open=False):
                detailed_table = gr.Dataframe(label="Complete Analysis Results", wrap=True, visible=False)
                with gr.Row():
                    csv_download = gr.File(label="⬇️ Download CSV", visible=False)
                    json_download = gr.File(label="⬇️ Download JSON", visible=False)
            
            def handle_analysis_and_downloads(game_choice, progress=gr.Progress()):
                # Pass the progress tracker to the analysis function
                status, summary_df, full_df = analyze_all_reviews(game_choice, progress)
                
                if full_df is not None:
                    csv_path = "reviews_analysis.csv"
                    json_path = "reviews_analysis.json"
                    
                    full_df.to_csv(csv_path, index=False)
                    full_df.to_json(json_path, orient='records', indent=4)
                    
                    return (
                        status,
                        summary_df,
                        full_df, # For the hidden detailed view
                        gr.File(value=csv_path, visible=True),
                        gr.File(value=json_path, visible=True)
                    )
                else:
                    return (
                        status,
                        None,
                        None,
                        gr.File(visible=False),
                        gr.File(visible=False)
                    )
            
            scrape_btn.click(
                fn=handle_analysis_and_downloads,
                inputs=[game_dropdown],
                outputs=[status_output, summary_table, detailed_table, csv_download, json_download]
            )
        
        with gr.TabItem("📘 Q&A"):
            gr.Markdown("""
            ## 📘 Technical Q&A
            
            ### **🤖 Which NLP models or libraries did you use for review analysis and why?**
            
            **Primary Models:**
            - **Sentiment Analysis:** `tabularisai/multilingual-sentiment-analysis` (RoBERTa-based)
                - Supports 19+ languages including Turkish, Arabic, Spanish
                - 5-level granularity: Very Negative → Very Positive
                - Higher accuracy than generic sentiment models
            
            - **Zero-Shot Classification:** `cross-encoder/nli-distilroberta-base`
                - Used for both fake detection AND interesting review detection
                - Efficient single model for multiple classification tasks
                - No training data required, just define candidate labels
            
            - **Semantic Similarity:** `all-MiniLM-L6-v2` (SentenceTransformers)
                - Fast 384-dim embeddings for duplicate detection
                - Cosine similarity threshold of 0.95 for near-duplicates
                - Only 22MB model size for quick inference
            
            **Why these choices?**
            - **Offline-first:** No API dependencies or costs
            - **Lightweight:** Combined models < 500MB RAM usage
            - **Multilingual:** Critical for global app store reviews
            - **GPU optimized:** CUDA support with FP16 precision
            
            ### **🕵️ What strategy did you follow for fake review detection?**
            
            **Hybrid 3-Layer Detection System:**
            
            **1. Heuristic Analysis (Rule-based):**
            ```python
            # Text complexity features
            unique_word_ratio = unique_words / total_words
            entropy = -sum((count/total) * log2(count/total))
            personal_pronoun_count = count("I", "me", "my", "we")
            url_detection = regex_search(r'http|www|\.com')
            
            # Anomaly scoring
            if unique_word_ratio < 0.4 and word_count > 15:
                anomaly_score += 0.4  # Repetitive language
            if entropy < 2.5 and word_count > 10:
                anomaly_score += 0.3  # Low complexity
            if url_count > 0:
                anomaly_score = 1.0   # Immediate spam flag
            ```
            
            **2. Semantic Duplicate Detection (Embedding-based):**
            ```python
            current_embedding = similarity_model.encode(text)
            for prev_embedding in stored_embeddings:
                similarity = cosine_similarity(current, prev)
                if similarity > 0.95:
                    return {"is_fake": True, "label": "duplicate"}
            ```
            
            **3. ML Classification (Zero-shot):**
            ```python
            labels = ["promotional spam", "uninformative review", 
                     "genuine personal story", "constructive feedback"]
            result = classifier(text, labels)
            if result['labels'][0] in ['promotional spam'] and score > 0.8:
                return fake_classification
            ```
            
            **Final Decision Logic:**
            `final_confidence = max(anomaly_score, zero_shot_score)`
            
            ### **😊 How did you compute sentiment scores?**
            
            **Primary Method - TabularisAI Multilingual Model:**
            ```python
            # Tokenize and predict
            inputs = tokenizer(text, max_length=512, truncation=True)
            outputs = model(**inputs)
            probabilities = softmax(outputs.logits)
            
            # 5-level mapping
            sentiment_map = {
                0: "Very Negative", 1: "Negative", 2: "Neutral", 
                3: "Positive", 4: "Very Positive"
            }
            
            # Convert to polarity score (-1.0 to +1.0)
            if sentiment in ["Very Positive", "Positive"]:
                polarity = confidence * (1.0 if "Very" else 0.7)
            elif sentiment in ["Very Negative", "Negative"]:
                polarity = -confidence * (1.0 if "Very" else 0.7)
            ```
            
            **Fallback Methods:**
            1. **Zero-shot classifier:** 3-way classification (positive/neutral/negative)
            2. **Lexicon-based:** Word counting with predefined positive/negative lists
            
            **Output Format:**
            - Simplified sentiment: Positive/Negative/Neutral
            - Detailed sentiment: Very Positive, Positive, etc.
            - Polarity score: -1.0 (very negative) to +1.0 (very positive)
            - Confidence score: Model's certainty in prediction
            - Raw scores: All 5 probability distributions
            
            ### **⭐ What logic did you use to select interesting reviews?**
            
            **Automatic Detection via Zero-Shot Classification:**
            ```python
            candidate_labels = [
                "creative, funny, or detailed personal story",  # TARGET
                "generic, standard comment",
                "sarcastic or ironic",
                "overly enthusiastic or exaggerated",
                "brief but informative",
                "repetitive or templated",
                "off-topic or unrelated",
                "technical or bug report style",
                "question or support request",
                "angry or ranting tone"
            ]
            
            result = classifier(text[:512], candidate_labels)
            is_interesting = (result['labels'][0] == candidate_labels[0] 
                            and result['scores'][0] > 0.6)
            ```
            
            **Examples of Detected Interesting Reviews:**
            - **Storytelling:** "I was playing at 3 AM when my cat jumped on my phone and somehow got me to the secret level..."
            - **Creative:** "An absolute masterpiece reminiscent of classic literature..."
            - **Humorous:** "I may have done a victory dance that woke up the entire apartment building..."
            
            **Why This Works:**
            - Captures narrative structure and personal anecdotes
            - Identifies creative language and storytelling elements
            - Filters out generic "good game" comments automatically
            
            ### **🔄 How did you implement review scraping? Is the data continuously updatable?**
            
            **Implementation:**
            ```python
            from google_play_scraper import reviews, app
            
            def scrape_reviews(app_id, count=500):
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,
                    count=count
                )
                return result
            ```
            
            **Data Update Strategy:**
            - **Manual trigger:** Users can re-scrape anytime via Gradio interface
            - **Fresh data:** Always pulls latest reviews from Google Play
            - **No persistence:** Currently stateless (could add database)
            - **Rate limiting:** Google Play has implicit rate limits (~1000 reviews/request)
            
            **Potential Automation:**
            ```python
            # Could add scheduled scraping
            import schedule
            schedule.every(6).hours.do(scrape_and_analyze_all_games)
            ```
            
            ### **🚀 If you had to make this scalable and real-time, how would you architect the system?**
            
            **Current State:** Single-machine Gradio app (good for demo/prototyping)
            
            **Production Architecture (As a recent grad perspective):**
            
            **Phase 1 - Basic Production (0-1K users):**
            ```
            Frontend: React/Vue.js
            Backend: FastAPI/Flask
            Database: PostgreSQL
            Cache: Redis
            Deployment: Docker → Heroku/Railway
            ```
            
            **Phase 2 - Scalable (1K-100K users):**
            ```
            Load Balancer (nginx)
            ├── API Servers (3x FastAPI instances)
            ├── Background Workers (Celery + Redis)
            │   ├── Review Scraping Jobs
            │   ├── ML Analysis Jobs  
            │   └── Batch Processing
            ├── Database Cluster (PostgreSQL + Read Replicas)
            ├── Model Serving (TorchServe/TensorFlow Serving)
            └── Monitoring (Prometheus + Grafana)
            ```
            
            **Phase 3 - Enterprise (100K+ users):**
            ```
            Microservices Architecture:
            ├── API Gateway (Kong/AWS API Gateway)
            ├── Review Scraper Service
            ├── ML Analysis Service  
            ├── Real-time Processing (Apache Kafka)
            ├── Data Lake (AWS S3/Snowflake)
            └── Auto-scaling (Kubernetes)
            ```
            
            **Real-time Features:**
            - **WebSocket connections** for live analysis updates
            - **Stream processing** with Apache Kafka/Pulsar
            - **Model caching** with Redis for frequent queries
            - **CDN** for static assets and cached results
            
            **Key Technologies to Learn:**
            - **Backend:** FastAPI, async/await, database design
            - **Infrastructure:** Docker, Kubernetes, AWS/GCP basics
            - **Monitoring:** Logging, metrics, error tracking
            - **ML Ops:** Model versioning, A/B testing, monitoring drift
            
            **Honest Assessment (Recent Grad):**
            I'm still learning production systems, but I understand the concepts through coursework and personal projects. I'd start simple (Phase 1) and gradually scale based on actual user needs rather than over-engineering early. The key is measuring performance bottlenecks and scaling the specific components that need it.
            """)

if __name__ == "__main__":
    app.launch(share=True, debug=True)