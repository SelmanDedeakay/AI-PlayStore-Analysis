# 🎮 AI PlayStore Analysis - Technical Documentation

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/SelmanDedeakay/AI-PlayStore-Analysis)

> **Repository:** https://github.com/SelmanDedeakay/AI-PlayStore-Analysis

## 🏗️ Architecture Overview

This project implements an AI-powered review analysis system for Google Play Store games using a hybrid approach combining rule-based heuristics, machine learning models, and semantic analysis.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │ → │   Analyzer.py    │ → │  ML Models      │
│   (Frontend)    │    │   (Core Logic)   │    │  (Transformers) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↑                       ↑                       ↑
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   App.py        │    │   Scraper.py     │    │  GPU/CPU        │
│   (Interface)   │    │   (Data Source)  │    │  (Hardware)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🤖 Machine Learning Models

### **Sentiment Analysis**
- **Primary:** `tabularisai/multilingual-sentiment-analysis` (RoBERTa-based)
- **Features:** 5-level sentiment classification, 19+ language support
- **Output:** Polarity score (-1.0 to +1.0), confidence, detailed sentiment

### **Zero-Shot Classification**
- **Model:** `cross-encoder/nli-distilroberta-base`
- **Use Cases:** Fake detection, interesting review detection
- **Advantage:** No training data required, flexible label definitions

### **Semantic Similarity**
- **Model:** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Purpose:** Duplicate review detection
- **Method:** Cosine similarity with 0.95 threshold

## 🕵️ Fake Review Detection System

### **3-Layer Hybrid Approach:**

**Layer 1: Heuristic Analysis**
```python
features = {
    'unique_word_ratio': unique_words / total_words,
    'entropy': -sum((count/total) * log2(count/total)),
    'personal_pronoun_count': count("I", "me", "my", "we"),
    'url_count': regex_count(r'http|www|\.com')
}
```

**Layer 2: Semantic Duplicate Detection**
```python
similarity = cosine_similarity(current_embedding, stored_embeddings)
if similarity > 0.95:
    return {"is_fake": True, "label": "duplicate"}
```

**Layer 3: ML Classification**
```python
labels = ["promotional spam", "uninformative review", 
          "genuine personal story", "constructive feedback"]
confidence = classifier(text, labels)
```

## ⭐ Interesting Review Detection

Uses zero-shot classification with 10 candidate labels:
- **Target:** "creative, funny, or detailed personal story"
- **Threshold:** 0.6 confidence score
- **Examples:** Storytelling, humor, detailed feedback

```python
candidate_labels = [
    "creative, funny, or detailed personal story",  # TARGET
    "generic, standard comment",
    "sarcastic or ironic",
    # ... 7 more labels
]
```

## 🔄 Data Pipeline

### **Review Scraping**
```python
from google_play_scraper import reviews_all
import pandas as pd
from tqdm import tqdm

def scrape_reviews(app_id, langs=None):
    if langs is None:
        # Support for 70+ languages
        langs = ["en", "tr", "es", "de", "fr", "ru", "pt", "ar", 
                "zh", "hi", "ja", "ko", "id", "it", "vi", "th", ...]
    
    all_reviews = []
    
    for lang in tqdm(langs, desc="Languages"):
        try:
            reviews = reviews_all(
                app_id,
                lang=lang,
                country='us',
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
            print(f"Error scraping lang={lang}: {e}")
    
    return all_reviews
```

### **Processing Flow**
1. **Scrape** → Google Play Store API
2. **Analyze** → ML models (fake, sentiment, interesting)
3. **Cache** → Memory-based caching with hashing
4. **Export** → CSV/JSON formats

## 🚀 Performance Optimizations

### **Memory Management**
- GPU memory cleanup after each analysis
- LRU caching for repeated text analysis
- Embedding storage on CPU to save VRAM
- FP16 precision for GPU inference

### **Caching Strategy**
```python
@lru_cache(maxsize=1000)
def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()
```

### **Hardware Optimization**
- **CUDA Support:** Automatic GPU detection and utilization
- **Mixed Precision:** FP16 for memory efficiency
- **Batch Processing:** Efficient tensor operations

## 📊 Technical Specifications

| Component | Technology | Size | Purpose |
|-----------|------------|------|---------|
| Sentiment Model | TabularisAI RoBERTa | ~500MB | Multilingual sentiment |
| Zero-shot Classifier | DistilRoBERTa | ~250MB | Fake/Interesting detection |
| Similarity Model | MiniLM-L6-v2 | ~22MB | Duplicate detection |
| **Total Memory** | **Combined** | **<1GB** | **All models loaded** |

## 🛠️ Installation & Setup

### **Requirements**
```bash
pip install -r requirements.txt
```

### **Key Dependencies**
- `transformers` - HuggingFace models
- `torch` - PyTorch backend
- `sentence-transformers` - Semantic similarity
- `google-play-scraper` - Data source
- `gradio` - Web interface

### **Quick Start**
```bash
python app.py  # Launch Gradio interface
python analyzer.py  # Run standalone tests
```

## 📈 Scalability Considerations

### **Current Limitations**
- Single-threaded processing
- Memory-based caching (not persistent)
- Limited to 50 reviews per analysis (demo)

### **Production Improvements**
- **Database:** PostgreSQL for persistent storage
- **Queue System:** Celery + Redis for background processing
- **API:** FastAPI for REST endpoints
- **Containerization:** Docker deployment
- **Monitoring:** Logging and metrics collection

## 🛡️ Error Handling

### **Graceful Degradation**
- Sentiment analysis falls back to lexicon-based if model fails
- Fake detection uses heuristics if ML models unavailable
- Comprehensive exception handling with user-friendly messages


---

**Repository:** https://github.com/SelmanDedeakay/AI-PlayStore-Analysis

