# 🎮 AI-Powered Play Store Review Analysis System

Note: Please see the demo video first --> https://drive.google.com/file/d/19be7r07Kb0hb2haPpoYb0lNjqqGhdM2W/view?usp=sharing

A comprehensive AI-powered system for analyzing Google Play Store reviews with advanced NLP capabilities including fake review detection, sentiment analysis, and interesting content identification.

## 🚀 Features

- **🕵️ Fake Review Detection**: Hybrid approach combining heuristics and transformer models
- **😊 Sentiment Analysis**: Multilingual sentiment analysis with 5-level classification
- **⭐ Interesting Review Detection**: Automatic identification of creative, funny, and detailed reviews
- **🔍 Duplicate Detection**: Semantic similarity-based duplicate review identification
- **🌍 Multilingual Support**: Supports 19+ languages for sentiment analysis
- **📊 Interactive Web Interface**: Gradio-based GUI for easy analysis
- **💾 Export Capabilities**: CSV and JSON export options

## 🛠️ Technology Stack

### Core NLP Models
- **Sentiment Analysis**: `tabularisai/multilingual-sentiment-analysis` (Multilingual RoBERTa)
- **Classification**: `cross-encoder/nli-distilroberta-base` (Zero-shot classification)
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers for similarity)

### Frameworks & Libraries
- **Web Interface**: Gradio
- **ML/NLP**: Transformers, PyTorch, Sentence Transformers
- **Data Processing**: Pandas, NumPy
- **Review Scraping**: google-play-scraper

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Internet connection for model downloads

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-PlayStore-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (automatic on first run)
   - Models will be downloaded automatically when first used
   - Ensure stable internet connection for initial setup

## 🚀 Usage

1. **Launch the Gradio interface**
   ```bash
   python app.py
   ```

2. **Access the interface**
   - Local: `http://localhost:7860`
   - Public link will be provided in console

3. **Use the interface**
   - **Manual Review Tab**: Analyze individual reviews
   - **Full Review Analysis Tab**: Scrape and analyze all reviews for a game
   - **Q&A Tab**: Learn about the system architecture

## 📊 Workflow

### 1. Review Collection
- Uses `google-play-scraper` library for efficient data collection
- Supports multiple games and languages
- Collects comprehensive metadata (ratings, dates, helpful votes)

### 2. AI Analysis Pipeline

```
Raw Review Text
       ↓
┌─────────────────────────────────────────┐
│           Preprocessing                 │
│  • Text normalization                   │
│  • Language detection                   │
│  • Feature extraction                   │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│        Fake Review Detection           │
│  • Heuristic analysis                  │
│  • Duplicate checking                  │
│  • Zero-shot classification            │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│       Sentiment Analysis               │
│  • Multilingual RoBERTa model          │
│  • 5-level classification              │
│  • Polarity scoring                    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│    Interesting Content Detection       │
│  • Zero-shot classification            │
│  • Story/creativity identification     │
│  • Detailed feedback recognition       │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│          Results & Export              │
│  • Structured data output              │
│  • CSV/JSON export                     │
│  • Interactive visualization           │
└─────────────────────────────────────────┘
```

### 3. Output & Analysis
- Structured results with confidence scores
- Exportable data formats
- Real-time processing feedback

## 📁 Project Structure

```
AI-PlayStore-Analysis/
├── analyzer.py              # Core AI analysis engine
├── app.py                   # Gradio web interface
├── scraper.py               # Review collection module
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── output/                 # Analysis results
│   ├── final_reviews.csv
│   └── final_reviews.json
└── __pycache__/           # Python cache files
```

## 🎯 Supported Games

The system currently supports analysis for these games:
- Patrol Officer - Cop Simulator
- Desert Warrior
- Arcade Ball.io - Let's Bowl!
- Wrestling Trivia Run
- Chips Factory - Tycoon Game
- Deck Dash: Epic Card Battle RP
- Wedding Rush 3D!
- Hospital Life
- 1001 Brain Zen Puzzles
- Wand Evolution: Magic Mage Run
- Take'em Down!
- Top Race: Car Battle Racing
- Cross'em All
- Dog Whisperer: Fun Walker Game

## 🔧 Configuration

### Model Settings
Models are automatically configured for optimal performance:
- GPU acceleration when available
- Mixed precision for memory efficiency
- Intelligent caching for repeated analyses

### Memory Management
- Automatic GPU memory cleanup
- LRU caching for frequently analyzed texts
- Efficient batch processing

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Model Download Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python analyzer.py
   ```

3. **Network Errors**
   ```bash
   # Check proxy settings
   pip install --proxy http://proxy:port package_name
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Google** for Play Store data access
- **TabularisAI** for multilingual sentiment model
- **CardiffNLP** for NLP research contributions

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the Q&A section in the web interface
- Review the troubleshooting guide above

---

Made with ❤️ for better app review analysis
