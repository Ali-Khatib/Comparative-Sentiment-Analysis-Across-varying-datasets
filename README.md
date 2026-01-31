# Comparative Sentiment Analysis Across SST, IMDB, and Amazon Reviews

This repository presents a unified, large-scale sentiment analysis study across three widely used benchmark datasets. Multiple modeling paradigms are implemented and compared, ranging from lexicon-based methods to deep neural networks and transformer-based models.

---

## Datasets Used

### 1. Stanford Sentiment Treebank (SST)
Source: Stanford NLP  
Files:
- trainDevTestTrees_PTB/trees/train.txt
- trainDevTestTrees_PTB/trees/test.txt

Sentiment Labels (5-class):
0 – Very Negative  
1 – Negative  
2 – Neutral  
3 – Positive  
4 – Very Positive  

Tree-structured annotations are parsed, flattened, cleaned, and merged into a single dataset.

---

### 2. IMDB Movie Reviews
Source: Kaggle  
File:
- imdb-movie-ratings-sentiment-analysis/movie.csv

Sentiment Labels (Binary):
0 – Negative  
1 – Positive  

Used to evaluate binary sentiment classification on long-form reviews.

---

### 3. Amazon Customer Reviews Polarity
Source: Kaggle  
Files:
- amazon-customerreviews-polarity/train.csv
- amazon-customerreviews-polarity/test.csv

Sentiment Labels (Binary):
1 – Negative  
2 – Positive  

Train and test files are merged and a large random subset is sampled for efficiency.

---

## Preprocessing Pipeline (Shared Across Datasets)

- Lowercasing
- Regex-based punctuation and symbol removal
- Tokenization (NLTK)
- Lemmatization (WordNet)
- Stopword analysis
- Review length and word count extraction
- N-gram frequency analysis (uni-grams and bi-grams)

---

## Models Implemented

### Lexicon-Based
- **TextBlob**
  - Polarity-based sentiment scoring
- **VADER**
  - Negative / Neutral / Positive scoring
  - Custom threshold-based label mapping

### Classical Machine Learning
- **TF-IDF Vectorization**
  - Unigrams + Bigrams
  - Max features: 500
- **Logistic Regression**
- **Multinomial Naive Bayes**

### Transformer-Based
- **RoBERTa (j-hartmann/sentiment-roberta-large-english-3-classes)**
  - Local model loading (offline-compatible)
  - Batched inference for large datasets
  - Label alignment with dataset ground truth

### Deep Learning
- **1D Convolutional Neural Network**
  - Pretrained GloVe embeddings (100d)
  - CNN + GlobalMaxPooling architecture
  - Binary and 5-class variants
  - Implemented using TensorFlow / Keras

---

## Evaluation

- Accuracy
- Precision / Recall / F1-score
- Confusion Matrices
- Class distribution visualizations
- Polarity distribution plots
- Model-specific performance comparisons across datasets

---

## Visual Analysis

- Sentiment distribution histograms
- KDE plots for polarity vs sentiment
- Review length vs sentiment analysis
- Bigram frequency before/after stopword removal
- Word clouds (Amazon dataset)

---



## Key Takeaways

- Lexicon-based models are fast but biased toward positive sentiment.

- Classical ML models perform competitively with proper feature engineering.

- Transformer-based models generalize well across domains.

- CNNs benefit significantly from pretrained embeddings but require careful tuning.

## Environment

Python 3.9+

NLTK, Scikit-learn, TensorFlow

HuggingFace Transformers

Matplotlib / Seaborn

Kaggle Notebooks compatible
