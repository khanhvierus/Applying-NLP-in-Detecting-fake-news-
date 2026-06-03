# Fake News Classification System

## Overview
This project aims to develop a robust machine learning and deep learning system capable of accurately distinguishing between true and fake news articles. The solution addresses complex natural language processing (NLP) challenges by combining textual data with extracted statistical and categorical features.

## Dataset
The project utilizes the MisinfoSuperset dataset.
* The cleaned dataset contains a total of 68,604 well-balanced articles (50.3% true, 49.7% fake).
* True news articles are sourced from reputable media organizations such as Reuters and The New York Times.
* Fake news articles are gathered from extremist and propaganda websites.

## Detailed Project Workflow

### Phase 1: Initial Preprocessing & EDA
* **Data Integration & Cleaning:** Combined the true and fake datasets, and strictly removed 29 missing values and 10,012 duplicated records to prevent overfitting.
* **Exploratory Data Analysis (EDA):** Visualized article length distributions, identifying that fake news tends to contain more extreme outliers regarding text length.

### Phase 2: LLM-Assisted Feature Engineering
* **Subject Categorization:** Utilized the Mistral LLM API (`Pixtral-12B-2409`) to automatically classify each article's topic into one of 19 predefined categories (e.g., politics, worldNews, health).
* **Statistical Extraction:** Extracted 12 numerical features from the raw text, analyzing aspects like total word count, unique word ratio, uppercase words, and punctuation ratios (exclamation marks, question marks).

### Phase 3: Deep Cleaning & Formatting
* **Text Normalization:** Applied regular expressions to keep only alphabetic characters, converted all text to lowercase, and removed standard English stopwords to reduce noise.
* **Categorical Encoding:** Handled API response errors and applied one-hot encoding to convert the 19 subject categories into 18 binary dummy variables.

### Phase 4: Strict Data Splitting
* **Stratified Split:** Split the dataset into 80% training (54,883 samples) and 20% testing (13,721 samples) while maintaining the original true/fake class distribution.
* **Leakage Prevention:** This step was explicitly executed *before* any text vectorization to ensure the test set remained completely unseen by the models.

### Phase 5: Tokenization & Vectorization
* **TF-IDF Processing:** Applied `TfidfVectorizer` configured with an n-gram range of (1, 2) and a maximum of 25,000 features.
* **Feature Fusion:** Concatenated the resulting sparse text matrix with the 30 engineered numerical/categorical features, creating a comprehensive input array (25,030 features) for the models.

### Phase 6: Model Training & Hyperparameter Tuning
* **Traditional Machine Learning:** Trained Logistic Regression and LinearSVC as baseline models, utilizing Grid Search with 3-fold cross-validation across 336 parameter combinations.
* **Large Language Models (LLMs):** Fine-tuned BERT and XLNet using a custom hybrid architecture. This design combined 768-dimensional contextual embeddings with the 30 numerical features through dense layers.
* **Advanced Optimization:** Leveraged the Optuna framework, Mixed Precision Training (AMP), gradient accumulation, and early stopping to stabilize training and avoid overfitting.

### Phase 7: Evaluation
* Evaluated baseline traditional Machine Learning models including Logistic Regression and LinearSVC.
* Implemented and fine-tuned advanced Large Language Models, specifically BERT and XLNet. 
* Designed a hybrid architecture for the deep learning models to process both textual embeddings and numerical features simultaneously.

### Key Results
* **Logistic Regression:** Achieved 93.80% accuracy but showed signs of overfitting.
* **LinearSVC (Optimized):** Delivered 92.62% accuracy with excellent generalization and fast training times.
* **XLNet:** Reached 94.70% accuracy, providing a solid balance between performance and training efficiency.
* **BERT (Fine-tuned):** Achieved the highest overall performance with an accuracy of 95.59%.
---

## Workflow Diagram

```mermaid
graph LR
    A[Raw Datasets] --> B[Data Cleaning & EDA]
    B --> C[Feature Engineering via LLM & Statistics]
    C --> D[Text Cleaning & One-Hot Encoding]
    D --> E[Stratified Train/Test Split 80/20]
    E --> F[TF-IDF Vectorization]
    F --> G[Feature Concatenation]
    G --> H[Model Training & Optuna Tuning]
    H --> I[Evaluation & Final Output]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
