# Sentiment Analysis of Amazon Reviews

This project applies **traditional sentiment analysis** and **state-of-the-art NLP techniques** to Amazon food reviews, comparing the performance of both methods. By leveraging Python libraries and pre-trained deep learning models, it provides a comprehensive sentiment analysis framework for review classification.

---

## Features
- **Data Exploration and Visualization**  
  - Load and visualize Amazon food review data.  
  - Analyze distribution of review scores with matplotlib and seaborn.

- **Sentiment Analysis with VADER (NLTK)**  
  - Tokenizes and performs Part-of-Speech (POS) tagging.  
  - Extracts sentiment scores (positive, neutral, negative, and compound) for text.

- **Advanced Sentiment Analysis with RoBERTa**  
  - Utilizes Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment` model for fine-grained sentiment classification.  
  - Extracts probabilistic scores for negative, neutral, and positive sentiments.

- **Comparison of Approaches**  
  - Visualizes sentiment trends and correlations with review scores for both models.  
  - Compares the nuances captured by VADER and RoBERTa models.

---

## Libraries and Tools
- **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn, tqdm  
- **NLP Tools:** NLTK (VADER), Hugging Face Transformers  
- **Visualization:** Matplotlib, Seaborn  
- **Deep Learning Framework:** Hugging Face Transformers (RoBERTa)

---

## How It Works
1. **Data Loading**  
   Loads a subset of Amazon food reviews from a CSV file.

2. **Data Visualization**  
   Visualizes review score distribution to understand the dataset.

3. **Traditional Sentiment Analysis (VADER)**  
   - Tokenizes text and performs sentiment scoring.  
   - Outputs scores for positive, neutral, negative, and compound sentiments.

4. **Advanced Sentiment Analysis (RoBERTa)**  
   - Processes text using a pre-trained RoBERTa sentiment analysis model.  
   - Outputs probabilities for negative, neutral, and positive sentiment.

5. **Results Integration**  
   Combines sentiment scores with metadata for robust analysis.

6. **Visualization of Results**  
   - Displays compound sentiment trends for review scores.  
   - Compares sentiment dimensions (positive, neutral, negative) for different scores.

---

## Setup and Usage
### Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn, nltk, transformers, tqdm

### Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-reviews.git
   cd sentiment-analysis-reviews
2. Install the required packages using pip:
First, make sure you're in the project directory, then run:
    `pip install -r requirements.txt`

3. Download NLTK Data for VADER:
The script uses NLTK for sentiment analysis, so you'll need to download the necessary datasets by running the following Python commands:
```import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')`
```
4. Running the Project
To run the script, simply execute the Python file:
  `python sentiment_analysis.py`

### Results
**Insights into Sentiment Trends**
- The project demonstrates how advanced models like RoBERTa provide more nuanced sentiment analysis compared to traditional tools.

**Improved Understanding**
- Visual comparisons highlight differences between models for diverse review scores.

