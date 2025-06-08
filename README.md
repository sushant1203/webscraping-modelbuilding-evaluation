# Python Recipe Recommender & Predictive Model

## 📖 Project Overview

This is an end-to-end data science project that demonstrates a full data pipeline, from data acquisition and feature engineering to model development and evaluation. The project's core objectives were to build a content-based recipe recommender system and a predictive model to classify recipe "tastiness."

The project showcases skills in web scraping, data pre-processing, natural language processing techniques for feature extraction, and the implementation and evaluation of machine learning models.

---

## ✨ Key Features & Technical Breakdown

### 1. Data Acquisition (Web Scraping)
* Developed a Python script using **BeautifulSoup** to scrape recipe data from a dynamic website.
* The scraped data, including recipe titles, ratings, and ingredients, was parsed and structured into a Pandas DataFrame.
* The final clean dataset was exported to a `.csv` file for use in the modelling stages.

### 2. Content-Based Recommender Engine
* Engineered a `combine_features` column by concatenating key textual features (`title`, `category`, `ingredients`, etc.) to create a "document" for each recipe.
* Utilised **`CountVectorizer`** from Scikit-learn to convert these text documents into a matrix of token counts.
* Computed a **cosine similarity matrix** to determine the similarity between all pairs of recipes based on their content.
* Implemented two distinct recommendation functions:
    * **Vector Space Method:** Returns the top 10 most similar recipes for a given input recipe using matrix-vector multiplication.
    * **K-Nearest Neighbors (KNN):** Implemented the KNN algorithm to find the 10 most similar recipes, offering an alternative approach to similarity calculation.

### 3. Predictive Model for Recipe "Tastiness"
* Developed a binary classification model to predict whether a recipe would be rated as "tasty" based on its features.
* The model was trained and tested on the dataset, and its performance was evaluated to determine its predictive accuracy.

---

## 🛠️ Skills & Technologies

* **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, BeautifulSoup
* **Data Science Concepts:** Web Scraping, Data Cleaning & Pre-processing, Feature Engineering
* **Machine Learning:**
    * **Natural Language Processing (NLP):** CountVectorizer, Cosine Similarity
    * **Recommender Systems:** Content-Based Filtering, Vector Space Model, K-Nearest Neighbors (KNN)
    * **Predictive Modeling:** Classification, Model Training & Evaluation

---

## 🤝 My Contributions

* I was responsible for developing the web scraping module using BeautifulSoup to collect the initial dataset.
* I implemented the feature engineering and the cosine similarity matrix using Scikit-learn.
* I developed and evaluated the predictive model for classifying recipe tastiness.

---

## 📄 License

* © [2025] [Sushant Jasra Kumar] All Rights Reserved.
