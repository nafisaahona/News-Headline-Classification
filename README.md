# News Headline Classification

A large-scale NLP project benchmarking 8 model architectures across multiple text representations and preprocessing strategies for multi-class news headline classification, in PyTorch (CUDA).

## Overview

Classifies news headlines into 4 categories — **Business**, **Science & Technology**, **Sports**, **World News** — using a systematic comparison of classical ML, deep learning, and recurrent architectures over two text representations.

## Dataset

- 104,763 training records (89,048 train / 15,715 validation, stratified split) + a held-out 12,000-sample test set
- Class distribution is imbalanced (Science & Technology and Sports are majority classes)

## Pipeline

**Preprocessing** — three variants tested:
- **Raw** — unmodified text (baseline)
- **Extreme** — HTML/punctuation removal, lowercasing, alphabetic-only tokens, stopword removal, Porter stemming
- **Optimum** — balanced cleaning (HTML/punctuation removal, lowercasing) while preserving stopwords and numbers, which can carry useful signal in short headlines

**Text representation:**
- **TF-IDF** (unigram + bigram) — used with Logistic Regression and a Deep Neural Network
- **Skip-gram Word2Vec** — self-trained dense embeddings, used to initialize the embedding layer of sequence models

**Models trained** (24+ combinations across preprocessing × representation):
- Logistic Regression, Deep Neural Network
- SimpleRNN, GRU, LSTM, and their bidirectional variants (BiSimpleRNN, BiGRU, BiLSTM)

## Results

| Representation | Model | Avg. Test Macro-F1 |
|---|---|---|
| Skip-gram | BiGRU | 91.62% |
| Skip-gram | BiLSTM | 91.56% |
| Skip-gram | GRU | 91.36% |
| Skip-gram | LSTM | 91.21% |
| TF-IDF | Deep NN | 90.66% |
| TF-IDF | Logistic Regression | 90.44% |

**Best individual run:** GRU + Skip-gram Word2Vec + optimum preprocessing → **91.89% test accuracy / 91.88% test macro-F1**.

Gated recurrent models (GRU/LSTM and bidirectional variants) consistently outperformed SimpleRNN and were the strongest sequence-based approach overall, while TF-IDF-based models remained strong, efficient baselines.

## Tech Stack

Python, PyTorch, CUDA, scikit-learn, NLTK/spaCy (preprocessing), matplotlib, seaborn

## How to Run

1. Open `CSE440_News_Project_Lab_PyTorchCUDA.ipynb` in a GPU-enabled environment (Colab/Kaggle/local CUDA).
2. Run all cells to reproduce preprocessing, embedding training, model training, and evaluation across all combinations.

## Report

Full methodology, EDA (word clouds, unigram analysis), and results are documented in the accompanying report.

## Notes

Originally built as a course project for CSE440. Limitation: models were trained only on headline text, which may lack sufficient context to disambiguate closely related topics.
