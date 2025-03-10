# Amazon Spam Review Classification

This repository contains the implementation of a machine learning-based system to detect spam reviews on e-commerce platforms. The project leverages multiple classification models and various text preprocessing techniques to build an effective spam detection pipeline.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
6. [Model Training & Evaluation](#model-training--evaluation)  
7. [Results](#results)  
---

## Project Overview

Spam reviews on e-commerce platforms mislead customers and affect businesses’ credibility. This project aims to develop a robust classification model using supervised machine learning techniques to detect and filter out spam reviews. The system evaluates multiple models, including:

- Logistic Regression  
- Multinomial Naive Bayes  
- Linear Support Vector Classifier (Linear SVC)  
- Perceptron  
- XGBoost  

The models are trained and tested using a publicly available dataset of Amazon product reviews labeled as spam and non-spam.

---

## Features

- **Data Preprocessing:**  
  - Cleaning: Removal of punctuation, lowercasing, stopword removal, and special character removal  
  - Lemmatization: Standardizing words to their base form  
  - Multiple versions of processed data (Original, Cleaned, Lemmatized, Cleaned & Lemmatized)  

- **Feature Engineering:**  
  - TF-IDF Vectorization  
  - Handling class imbalance through undersampling  

- **Model Training & Evaluation:**  
  - Evaluation of multiple ML models  
  - Performance comparison based on accuracy, precision, recall, and F1-score  
  - Confusion matrix analysis  

- **Visualization:**  
  - Word clouds for spam and non-spam reviews  
  - Distribution graphs of review counts over time  
  - Performance comparison of different models  

---

## Dataset

We used a dataset from Kaggle:  
[**Amazon Product Review Spam and Non-Spam Dataset**](https://www.kaggle.com/datasets/naveedhn/amazon-product-review-spam-and-non-spam)

### Dataset Details:
- 15.4 million reviewers  
- 26.7 million reviews  
- Binary class labels:  
  - Spam (1)  
  - Non-spam (0)  

### Key Features:
- **reviewText:** The review content  
- **overall:** The star rating given by the reviewer  
- **helpful:** Number of helpful votes  
- **category:** Product category  
- **reviewTime:** Date of the review  
- **class:** Spam (1) or non-spam (0)  

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.8+ installed along with the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib wordcloud nltk xgboost
```

### Cloning the Repository

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### Download Dataset

Download the dataset from Kaggle and place it in the `data/` directory.

---

## Usage

### 1. Data Preprocessing

Run the following script to clean and preprocess the data:

```bash
python preprocessing.py
```

This script performs:
- Text cleaning  
- Lemmatization  
- TF-IDF Vectorization  
- Data splitting (80-10-10 for training, testing, and validation)  

### 2. Model Training & Evaluation

To train and evaluate models, run:

```bash
python train_models.py
```

This script:
- Trains multiple ML models  
- Evaluates them using accuracy, precision, recall, and F1-score  
- Generates confusion matrices and ROC curves  

### 3. Model Performance Comparison

To compare the performance of models on different versions of processed data, execute:

```bash
python compare_models.py
```

This script provides insights into which model and preprocessing method yield the best spam detection performance.

---

## Model Training & Evaluation

The following models were trained and evaluated:

| Model                       | Accuracy | Precision | Recall  | F1-Score |
|-----------------------------|----------|-----------|---------|----------|
| Logistic Regression         | 90.99%   | 90.51%    | 90.99%  | 90.54%   |
| Multinomial Naive Bayes     | 86.69%   | 87.21%    | 86.69%  | 83.18%   |
| XGBoost                     | 91.02%   | 90.53%    | 91.02%  | 90.50%   |
| Perceptron                  | 87.63%   | 87.12%    | 87.63%  | 87.33%   |
| Linear SVC                  | 90.46%   | 89.91%    | 90.46%  | 89.77%   |

### Key Findings:
- **XGBoost with Cleaned & Lemmatized Data performed the best (Accuracy: 91.02%).**  
- **Logistic Regression and Linear SVC were close contenders.**  
- **Naive Bayes performed the worst due to its assumption of feature independence.**  

---

## Results

- **Data Cleaning & Preprocessing:**  
  - Word clouds indicate frequent words in spam vs. non-spam reviews.  
  - Review distribution analysis shows an increasing trend in spam over time.  

- **Feature Engineering:**  
  - TF-IDF vectorization improved model performance.  
  - Lemmatization helped reduce redundant features.  

- **Model Performance:**  
  - XGBoost outperformed other models, achieving the highest accuracy and recall.  
  - Logistic Regression & Linear SVC were effective but slightly behind XGBoost.  
  - Undersampling improved the model’s ability to detect spam by balancing class distribution.  

---

We welcome contributions! Follow these steps to contribute:  

1. **Fork the repository.**  
2. **Create a new branch** (`feature-new-idea`).  
3. **Commit your changes.**  
4. **Push to your fork and submit a Pull Request.**  
