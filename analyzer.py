import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

# Load sentiment model
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

_seen_reviews = set()
_last_n_reviews = []

def parse_llm_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    json_str = match.group()
    return json.loads(json_str)

# 1. Sentiment Analysis
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = softmax(outputs.logits[0].numpy())
    stars = probs.argmax() + 1
    if stars <= 2:
        label = "negative"
    elif stars == 3:
        label = "neutral"
    else:
        label = "positive"
    return {
        "label": label,
        "score": float(probs[stars - 1])
    }

# 2. Fake Review Detection – Gemini Destekli
def is_fake_review(text):
    text_norm = text.strip().lower()
    if text_norm in _seen_reviews:
        return True, "Exact duplicate"
    _seen_reviews.add(text_norm)

    global _last_n_reviews
    if len(_last_n_reviews) >= 5:
        corpus = _last_n_reviews + [text]
        vec = TfidfVectorizer().fit_transform(corpus)
        sim_matrix = cosine_similarity(vec[-1], vec[:-1])
        max_sim = sim_matrix.max()
        if max_sim > 0.95:
            return True, f"Near-duplicate (sim={max_sim:.2f})"
        _last_n_reviews = _last_n_reviews[1:]
    _last_n_reviews.append(text)

    prompt = f"""
You are analyzing mobile app reviews.

Determine whether the following review is likely to be fake, spam, low-effort, or AI-generated.

REVIEW:
{text}

Respond in this exact JSON format: {{"is_fake": true/false, "reason": "your explanation here"}}
    """
    try:
        res = model.generate_content(prompt)
        raw = res.text.strip()
        json_obj = parse_llm_response(raw)
        return json_obj["is_fake"], json_obj["reason"]
    except Exception as e:
        return False, f"LLM Error: {e}"

# 3. Interesting Review Detection – Gemini Destekli
def is_interesting_review(text):
    prompt = f"""
You are a review analysis agent. Determine whether the following review contains interesting content. 
"Interesting" means anything that's humorous, creative, unusual, detailed, constructive, or suggestive.

REVIEW:
{text}

Respond in this exact JSON format: {{"is_interesting": true/false, "reason": "your explanation"}}
    """
    try:
        res = model.generate_content(prompt)
        raw = res.text.strip()
        json_obj = parse_llm_response(raw)
        return json_obj["is_interesting"], json_obj["reason"]
    except Exception as e:
        return False, f"LLM Error: {e}"
