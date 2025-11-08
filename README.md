# Hybrid-Vector-Search-and-Retrieval-System-BigBasket-dataset-
This repository contains the complete workflow for analysing the BigBasket product dataset and developing a robust, high-speed hybrid recommendation engine. The project integrates traditional machine learning with modern deep learning techniques (BERT) and vector databases (FAISS) to offer sophisticated data insights and accurate product recommendations.

## 🌟 Key Features
- **Comprehensive Data Processing:** Detailed cleaning and preparation of the raw product data.
- **In-Depth Exploratory Data Analysis (EDA):** Visualisation and statistical analysis of product attributes (e.g., categories, prices, brands).
- **Hybrid Feature Classification:** A machine learning pipeline combining BERT embeddings (for product description text) with structured features (numerical/categorical data) to predict a key product attribute.
- **High-Speed Recommendation System:** Implements a Hybrid Search function using FAISS (Facebook AI Similarity Search) for blazing-fast vector lookups, integrated with semantic and rule-based filters.

## ⚙️ Libraries
- Data Handling:	pandas, numpy, pathlib
- Visualisation:	matplotlib.pyplot, seaborn
- Machine Learning:	sklearn, faiss
- Deep Learning/NLP:	torch, transformers, sentence_transformers

## 📁 Dataset
The data used for this project is the BigBasket Entire Product List (28K datapoints), available on Kaggle:
Source: https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints/data

## 🔬 Project Structure & Methodology
**1. Data Cleaning & Preprocessing**

Handles missing values, performs text normalisation, and encodes categorical variables.

**2. EDA & Visualisation**

Statistical summaries and visualisations to understand product distribution, pricing, and brand performance.

**3. Machine Learning**

This section focuses on using Principal Component Analysis (PCA) for dimensionality reduction prior to classification.

- Feature Selection: PCA was applied exclusively to a subset of pre-processed structured data, specifically:

  - category (One-Hot Encoded)
  - sale_price (Standard Scaled)
  - market_price (Standard Scaled)

- Preprocessing: Both price columns were standardised using StandardScaler to ensure features contributed equally to the variance calculation.

**4. Deep Learning**

This component develops a robust prediction model to automatically classify a new product's category and sub-category simultaneously.
- Multi-Task Goal: The model is trained simultaneously on two distinct classification tasks (Category and Sub-Category) using a single combined loss function to improve prediction accuracy across both.

- Hybrid Architecture: It leverages the power of BERT (bert-base-uncased) for product text embeddings, which are then concatenated with Standard Scaled Numerical features and One-Hot Encoded Categorical features.

- Data Pipeline: Product descriptions are tokenised via BertTokenizer, while all structured features are pre-processed using StandardScaler and OneHotEncoder before being loaded into the custom PyTorch ProductDataset.

- Key Functionality: This model provides a robust solution for inferring product attributes when new item details (text and structured data) are added to the BigBasket catalogue.

**5. Recommendation System**

- Vector Indexing: Product descriptions/names are converted to dense vectors using Sentence Transformer.

- FAISS Index: These vectors are indexed in FAISS for approximate nearest neighbour (ANN) search.

- Hybrid Search: A custom function is built to query the FAISS index (semantic match) and apply rule-based filters (e.g., category, brand, price range) for the final highly relevant recommendations.
