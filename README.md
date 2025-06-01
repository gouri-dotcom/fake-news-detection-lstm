# 📰 Fake News Detection using LSTM

A deep learning model built to classify news articles as real or fake using Natural Language Processing (NLP) techniques and LSTM architecture.

## 🔧 Tech Stack
- Python
- TensorFlow / Keras
- Natural Language Toolkit (NLTK)
- GloVe Pre-trained Word Embeddings

## 📁 Dataset
Used a dataset of 30,000+ news articles from Kaggle ([Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)).

## 🧠 Model Architecture
- Tokenization + Lemmatization
- Word Embedding using 100D GloVe vectors
- LSTM layers with dropout to reduce overfitting
- Dense output layer with sigmoid activation

## 📊 Results
- **Accuracy**: 91.2%
- **Evaluation**: Confusion Matrix, Precision, Recall, F1 Score

## 🔍 Example
| Metric        | Value   |
|---------------|---------|
| Accuracy      | 91.2%   |
| Precision     | 90.8%   |
| Recall        | 91.5%   |
| F1 Score      | 91.1%   |
