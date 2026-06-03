# Fake News Classification System

## Overview
This project aims to develop a robust machine learning and deep learning system capable of accurately distinguishing between true and fake news articles. The solution addresses complex natural language processing (NLP) challenges by combining textual data with extracted statistical and categorical features.

## Dataset
The project utilizes the MisinfoSuperset dataset.
* The cleaned dataset contains a total of 68,604 well-balanced articles (50.3% true, 49.7% fake).
* True news articles are sourced from reputable media organizations such as Reuters and The New York Times.
* Fake news articles are gathered from extremist and propaganda websites.

## Project Workflow

### 1. Data Preprocessing
* Removed missing values and duplicated records to ensure data quality.
* Cleaned the text data by standardizing to lowercase, removing special characters, and dropping stopwords.

### 2. Feature Engineering
* Extracted 12 statistical text features, including total word count, unique word ratio, and punctuation frequency.
* Utilized the Mistral LLM API (Pixtral-12B-2409) to automatically classify each article into one of 19 subject categories.

### 3. Data Splitting and Vectorization
* Split the dataset into training (80%) and testing (20%) sets prior to vectorization to strictly prevent data leakage.
* Applied TF-IDF vectorization with an n-gram range of (1, 2) and capped at 25,000 maximum features.

### 4. Model Training and Evaluation
* Evaluated baseline traditional Machine Learning models including Logistic Regression and LinearSVC.
* Implemented and fine-tuned advanced Large Language Models, specifically BERT and XLNet. 
* Designed a hybrid architecture for the deep learning models to process both textual embeddings and numerical features simultaneously.

## Key Results
* **Logistic Regression:** Achieved 93.80% accuracy but showed signs of overfitting.
* **LinearSVC (Optimized):** Delivered 92.62% accuracy with excellent generalization and fast training times.
* **XLNet:** Reached 94.70% accuracy, providing a solid balance between performance and training efficiency.
* **BERT (Fine-tuned):** Achieved the highest overall performance with an accuracy of 95.59%.
