# Shopping-AI
# 🛒 Online Shopping Purchase Prediction with KNN

This project is an AI model that predicts whether a user browsing an online shopping site will complete a purchase. Using a K-Nearest Neighbor classifier, the model analyzes session data such as page visits, browser usage, and whether the visit occurred on a weekend. It aims to improve both sensitivity and specificity for meaningful insights.

## 📚 Overview

The goal is to classify user purchasing intent based on session data, including:
- Page interactions
- Visitor type and traffic
- Browser and operating system
- Weekend activity
- Special occasions

Prediction results include:
- ✅ Correct predictions: 4088
- ❌ Incorrect predictions: 844
- 📈 True Positive Rate: 41.02%
- 📉 True Negative Rate: 90.55%

## 🚀 Getting Started

### Requirements

- Python 3.12
- scikit-learn

### Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/shopping-intent-classifier.git
   cd shopping-intent-classifier

   pip3 install scikit-learn

   python shopping.py shopping.csv


   shopping/
│
├── shopping.csv         # Source data with user session info
├── shopping.py          # Core implementation of the AI model
└── README.md            # Project overview and instructions

