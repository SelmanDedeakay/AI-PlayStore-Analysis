import hashlib
import json
import re
import gc
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import numpy as np
from scipy.special import softmax
import warnings
from collections import Counter
import math
from functools import lru_cache
from typing import Dict, Optional
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= GPU OPTIMIZATION =================
# Set memory management
torch.cuda.empty_cache()
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Use mixed precision for memory efficiency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = torch.cuda.is_available()
print(f"✅ Hardware setup: Running on {device} with FP16 precision: {use_fp16}")


# ================= LIGHTWEIGHT MODEL SETUP =================
# Lightweight embeddings model for similarity (part of hybrid fake detection)
from sentence_transformers import SentenceTransformer
try:
    _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    if torch.cuda.is_available(): _similarity_model = _similarity_model.to(device)
    print("✅ Similarity model loaded for duplicate detection.")
except Exception as e:
    print(f"Warning: Could not load sentence transformer, duplicate detection will be impaired: {e}")
    _similarity_model = None

# Lightweight Zero-Shot classifier for fake and interesting detection
_classifier_model_name = "cross-encoder/nli-distilroberta-base"
try:
    # This single pipeline will now handle both fake and interesting detection tasks
    _classifier_pipeline = pipeline("zero-shot-classification", model=_classifier_model_name, device=0 if torch.cuda.is_available() else -1)
    print(f"✅ Zero-shot classifier loaded: {_classifier_model_name}")
except Exception as e:
    print(f"⚠️ Could not load zero-shot classifier, fake & interesting detection will be impaired: {e}")
    _classifier_pipeline = None

# CardiffNLP Twitter RoBERTa sentiment model
_sentiment_model_name = "tabularisai/multilingual-sentiment-analysis"
try:
    _sentiment_tokenizer = AutoTokenizer.from_pretrained(_sentiment_model_name)
    _sentiment_model = AutoModelForSequenceClassification.from_pretrained(_sentiment_model_name)
    if torch.cuda.is_available(): 
        _sentiment_model = _sentiment_model.to(device)
    print(f"✅ Sentiment model loaded: {_sentiment_model_name}")
    
    # Sentiment mapping for the multilingual model
    _sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
except Exception as e:
    print(f"⚠️ Could not load sentiment model: {e}")
    _sentiment_tokenizer = None
    _sentiment_model = None
    _sentiment_map = None


# ================= ADVANCED CACHING SYSTEM =================
_seen_hashes = set()
_review_embeddings = []
_review_texts = []
_fake_cache = {}
_interesting_cache = {}
_sentiment_cache = {}

# Memory management decorator
def gpu_memory_cleanup(func):
    """Decorator to clean up GPU memory after function execution."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result
    return wrapper

@lru_cache(maxsize=1000)
def get_text_hash(text: str) -> str:
    """Generate hash for text caching."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

# ================= FEATURE CALCULATION =================
def calculate_advanced_text_features(text: str) -> Dict:
    """Calculates heuristic features from the text."""
    if not text: return {}
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    # Calculate entropy of the character distribution
    entropy = -sum((count/char_count) * math.log2(count/char_count) for count in Counter(text).values() if count > 0)
    features = {
        'word_count': word_count,
        'unique_word_ratio': len(set(words)) / max(word_count, 1),
        'entropy': entropy,
        'personal_pronoun_count': len(re.findall(r'(?i)\b(i|me|my|we|us|our)\b', text)),
        'url_count': len(re.findall(r'http[s]?://|www\.|\.[a-z]{2,4}', text.lower())),
    }
    return features

# ================= SIMILARITY CHECK =================
@gpu_memory_cleanup
def check_for_duplicates(text: str, threshold: float = 0.95) -> Optional[Dict]:
    """Checks if a review is semantically similar to recently seen ones."""
    if not _similarity_model or not text.strip(): return None
    current_embedding = _similarity_model.encode(text, convert_to_tensor=True, device=device)
    
    for prev_embedding, prev_text in zip(_review_embeddings, _review_texts):
        if text.strip() == prev_text.strip(): continue # Skip identical text
        
        # <<< FIX HERE >>> Move the previous embedding to the correct device before comparison
        similarity = torch.nn.functional.cosine_similarity(current_embedding.unsqueeze(0), prev_embedding.to(device).unsqueeze(0)).item()
        
        if similarity > threshold:
            return {"is_fake": True, "label": "duplicate", "confidence": similarity, "reason": f"Semantically similar to a previous review (score: {similarity:.2f})"}
    
    # Store embedding on CPU to save VRAM
    _review_embeddings.append(current_embedding.cpu())
    _review_texts.append(text)
    # Keep the cache size manageable
    if len(_review_embeddings) > 100:
        _review_embeddings.pop(0)
        _review_texts.pop(0)
    return None

# ================= FAKE REVIEW DETECTION =================
@gpu_memory_cleanup
def hybrid_fake_review_detection(text: str) -> Dict:
    """Detects fake reviews using a hybrid of heuristics and a zero-shot model."""
    text_hash = get_text_hash(text)
    if text_hash in _fake_cache: return _fake_cache[text_hash]

    if not text or len(text.strip().split()) < 3:
        return {"is_fake": True, "label": "low_effort", "confidence": 0.9, "reason": "Review is too short."}

    duplicate_result = check_for_duplicates(text)
    if duplicate_result:
        _fake_cache[text_hash] = duplicate_result
        return duplicate_result

    reasons, scores = [], {}
    features = calculate_advanced_text_features(text)
    anomaly_score = 0.0
    if features['unique_word_ratio'] < 0.4 and features['word_count'] > 15:
        anomaly_score += 0.4
        reasons.append("Very repetitive language.")
    if features['entropy'] < 2.5 and features['word_count'] > 10:
        anomaly_score += 0.3
        reasons.append("Low text complexity.")
    if features['personal_pronoun_count'] == 0 and features['word_count'] > 20:
        anomaly_score += 0.25
        reasons.append("Lacks personal pronouns, which can be a sign of non-genuine content.")
    if features['url_count'] > 0:
        anomaly_score = 1.0 # High penalty for URLs
        reasons.append("Contains a URL.")
    scores['anomaly'] = min(anomaly_score, 1.0)

    zero_shot_score = 0.0
    if _classifier_pipeline and anomaly_score < 0.9:
        try:
            candidate_labels = ["promotional spam", "uninformative review", "genuine personal story", "constructive feedback"]
            result = _classifier_pipeline(text[:512], candidate_labels, multi_label=False)
            is_spam = result['labels'][0] == 'promotional spam' and result['scores'][0] > 0.8
            is_uninformative = result['labels'][0] == 'uninformative review' and result['scores'][0] > 0.85
            if is_spam or is_uninformative:
                zero_shot_score = result['scores'][0]
                reasons.append(f"Classifier flags it as '{result['labels'][0]}'.")
            scores['zero_shot'] = zero_shot_score
        except Exception:
            scores['zero_shot'] = 0.0

    final_confidence = max(scores.get('anomaly', 0.0), scores.get('zero_shot', 0.0))
    is_fake = final_confidence > 0.7
    label = "genuine"
    if is_fake:
        label = "spam" if scores.get('anomaly', 0.0) >= 0.9 else "bot_generated" if scores.get('zero_shot', 0.0) > 0.8 else "low_effort"

    final_result = {"is_fake": is_fake, "label": label, "confidence": round(final_confidence, 3), "reason": reasons[0] if reasons else "Considered genuine."}
    _fake_cache[text_hash] = final_result
    return final_result

# ================= **NEW** EFFICIENT INTERESTING REVIEW DETECTION =================
@gpu_memory_cleanup
def detect_interesting_review(text: str, threshold: float = 0.6) -> Dict:
    """
    Detects if a review is 'interesting' using the lightweight zero-shot classifier.
    An interesting review tells a story, is creative, or provides detailed feedback.
    """
    text_hash = get_text_hash(text)
    if text_hash in _interesting_cache: return _interesting_cache[text_hash]

    if not _classifier_pipeline or len(text.strip().split()) < 10:
        return {"is_interesting": False, "label": "standard", "reason": "Review is too short for deep analysis."}
    
    # Define labels to classify the review's style
    candidate_labels = [
    "creative, funny, or detailed personal story",
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
    try:
        # Use the existing classifier pipeline
        result = _classifier_pipeline(text[:512], candidate_labels, multi_label=False)
        
        # Check if the top label is the "interesting" one and meets the confidence threshold
        is_interesting = result['labels'][0] == candidate_labels[0] and result['scores'][0] > threshold
        label = "interesting" if is_interesting else "standard"
        reason = f"Classifier Result: '{result['labels'][0]}' (Score: {result['scores'][0]:.2f})"
        
        final_result = {"is_interesting": is_interesting, "label": label, "reason": reason}

    except Exception as e:
        print(f"[Interesting Review Classifier Error] {e}")
        final_result = {"is_interesting": False, "label": "error", "reason": f"An error occurred during classification: {e}"}

    _interesting_cache[text_hash] = final_result
    return final_result

# ================= **NEW** SENTIMENT ANALYSIS =================
@gpu_memory_cleanup
def analyze_sentiment(text: str) -> Dict:
    """
    Analyzes sentiment of review text using TabularisAI multilingual sentiment model with fallback methods.
    Returns sentiment label (Very Negative/Negative/Neutral/Positive/Very Positive) and confidence score.
    """
    text_hash = get_text_hash(text)
    if text_hash in _sentiment_cache: 
        return _sentiment_cache[text_hash]

    if not text or len(text.strip().split()) < 2:
        return {"sentiment": "Neutral", "polarity": 0.0, "confidence": 0.0, "method": "default"}

    # Method 1: TabularisAI multilingual sentiment model (primary method)
    if _sentiment_model and _sentiment_tokenizer and _sentiment_map:
        try:
            # Tokenize and get model output
            inputs = _sentiment_tokenizer(text, return_tensors="pt", truncation=True, 
                                        padding=True, max_length=512)
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = _sentiment_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the predicted sentiment
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            sentiment_label = _sentiment_map[predicted_class]
            
            # Map to simplified sentiment and calculate polarity
            if sentiment_label in ["Very Positive", "Positive"]:
                simplified_sentiment = "Positive"
                polarity = confidence * (1.0 if sentiment_label == "Very Positive" else 0.7)
            elif sentiment_label in ["Very Negative", "Negative"]:
                simplified_sentiment = "Negative"
                polarity = -confidence * (1.0 if sentiment_label == "Very Negative" else 0.7)
            else:
                simplified_sentiment = "Neutral"
                polarity = 0.0
            
            # Get all probabilities for raw scores
            raw_scores = {
                _sentiment_map[i]: round(float(probabilities[0][i]), 4) 
                for i in range(len(_sentiment_map))
            }
            
            result = {
                "sentiment": simplified_sentiment,
                "detailed_sentiment": sentiment_label,
                "polarity": round(polarity, 3),
                "confidence": round(confidence, 3),
                "method": "tabularisai_multilingual",
                "raw_scores": raw_scores
            }
            
            _sentiment_cache[text_hash] = result
            return result
                    
        except Exception as e:
            print(f"[TabularisAI Sentiment Error] {e}")
    
    # Method 2: Fallback to transformers pipeline
    if _classifier_pipeline:
        try:
            candidate_labels = ["very positive review", "neutral or mixed review", "very negative review"]
            result = _classifier_pipeline(text[:512], candidate_labels, multi_label=False)
            
            # Map classifier result to sentiment
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            if "positive" in top_label:
                sentiment = "Positive"
                polarity = confidence * 0.8  # Scale to polarity range
            elif "negative" in top_label:
                sentiment = "Negative"
                polarity = -confidence * 0.8  # Negative polarity
            else:
                sentiment = "Neutral"
                polarity = 0.0
            
            fallback_result = {
                "sentiment": sentiment,
                "detailed_sentiment": sentiment,
                "polarity": round(polarity, 3),
                "confidence": round(confidence, 3),
                "method": "transformers_fallback"
            }
            
            _sentiment_cache[text_hash] = fallback_result
            return fallback_result
            
        except Exception as e:
            print(f"[Transformers Sentiment Error] {e}")
    
    # Method 3: Simple lexicon-based fallback
    positive_words = ["good", "great", "amazing", "love", "excellent", "awesome", "fantastic", "perfect", "best"]
    negative_words = ["bad", "terrible", "hate", "worst", "awful", "horrible", "disgusting", "boring", "sucks"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "Positive"
        polarity = min(pos_count * 0.3, 1.0)
    elif neg_count > pos_count:
        sentiment = "Negative"
        polarity = -min(neg_count * 0.3, 1.0)
    else:
        sentiment = "Neutral"
        polarity = 0.0
    
    confidence = min((abs(pos_count - neg_count) * 0.2), 1.0)
    
    lexicon_result = {
        "sentiment": sentiment,
        "detailed_sentiment": sentiment,
        "polarity": round(polarity, 3),
        "confidence": round(confidence, 3),
        "method": "lexicon_fallback"
    }
    
    _sentiment_cache[text_hash] = lexicon_result
    return lexicon_result

# ================= MAIN TEST FUNCTION =================
def test_analyzer():
    """Test the analyzer with the unified, efficient classifier."""
    test_reviews = [
        "This game is amazing! I love it so much.", # 0. Standard Positive
        "Click here for discount code! Limited time offer! www.fakegame.com", # 1. Spam with URL
        "I was playing this game late at night when my cat jumped on my phone and somehow got me to the secret level. Now she thinks she's a gaming pro! 10/10.", # 2. Humorous/Storytelling
        "So there I was, 3 AM, one life left, final boss battle. My roommate was asleep but I was screaming internally. When I finally beat it, I may have done a victory dance that woke up the entire apartment building. Worth it.", # 3. Storytelling
        "An absolute masterpiece of game design. The narrative weaves a complex tapestry of emotions, reminiscent of classic literature, while the gameplay itself feels both innovative and intuitive. A triumph.", # 4. Creative/Sophisticated
        "This game is really amazing! I love it so much.", # 5. Semantic Duplicate
    ]
    
    print("🚀 Efficient AI-Powered Review Analyzer (v5 - Lightweight Classifier)")
    print("=" * 80)
    
    for i, text in enumerate(test_reviews):
        print(f"\n{'='*80}\n📝 Review {i}/{len(test_reviews)-1}: {text[:80]}...")
        print("-" * 40)
        
        # Fake Detection
        print("🕵️ FAKE REVIEW DETECTION (HYBRID):")
        fake_result = hybrid_fake_review_detection(text)
        print(f"   • Status: {'🚨 FAKE' if fake_result['is_fake'] else '✅ GENUINE'}")
        print(f"   • Type: {fake_result['label']} (Confidence: {fake_result['confidence']})")
        
        # Interesting Detection
        print("\n🌟 INTERESTING REVIEW DETECTION (EFFICIENT CLASSIFIER):")
        if not fake_result['is_fake'] or fake_result['label'] not in ['spam', 'duplicate']:
            interesting_result = detect_interesting_review(text)
            print(f"   • Status: {'⭐ INTERESTING' if interesting_result['is_interesting'] else '📄 STANDARD'}")
            print(f"   • Reason: {interesting_result['reason']}")
        else:
            print("   • Skipped: Review identified as fake/spam.")
        
        # Sentiment Analysis
        print("\n😊 SENTIMENT ANALYSIS (TABULARISAI MULTILINGUAL + FALLBACK):")
        sentiment_result = analyze_sentiment(text)
        sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        print(f"   • Sentiment: {sentiment_emoji.get(sentiment_result['sentiment'], '😐')} {sentiment_result['sentiment']}")
        if 'detailed_sentiment' in sentiment_result:
            print(f"   • Detailed: {sentiment_result['detailed_sentiment']}")
        print(f"   • Polarity Score: {sentiment_result['polarity']} (Range: -1.0 to +1.0)")
        print(f"   • Confidence: {sentiment_result['confidence']}")
        print(f"   • Method: {sentiment_result['method']}")
        if 'raw_scores' in sentiment_result:
            print(f"   • Raw Scores: {sentiment_result['raw_scores']}")

# Memory cleanup utility
def cleanup_analyzer():
    """Clean up memory and caches."""
    global _review_embeddings, _review_texts, _fake_cache, _interesting_cache, _sentiment_cache
    _review_embeddings.clear(); _review_texts.clear(); _fake_cache.clear(); _interesting_cache.clear(); _sentiment_cache.clear()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    print("\n🧹 Memory and caches cleaned up")


if __name__ == "__main__":
    test_analyzer()
    cleanup_analyzer()