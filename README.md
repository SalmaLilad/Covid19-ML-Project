# Covid19-ML-Project
Using Machine Leaning models to predict COVID 19 from X-Rays.
# COVID-19 & Respiratory Disease Prediction

This repository contains a complete end-to-end machine learning project focused on predicting COVID-19 infection likelihood and extending the model to other respiratory ailments.

## 🔍 Overview
This project uses real-world clinical datasets (Kaggle and open medical data sources) to explore feature relationships, train multiple classification models, and analyze model performance.  
It demonstrates a full ML pipeline — from data collection to visualization, model training, and evaluation.

## 🧰 Technologies
- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, TensorFlow / PyTorch  
- Jupyter Notebooks for experimentation  
- GitHub for version control and collaboration  

## 🧠 Key Steps
1. **Data Preprocessing:** Cleaning missing values, handling outliers, and normalizing numeric data.  
2. **EDA (Exploratory Data Analysis):** Heatmaps, correlation matrices, and variable distributions.  
3. **Model Training:** Logistic Regression, Random Forest, XGBoost, CNN (for respiratory pattern data).  
4. **Evaluation:** Accuracy, F1-score, precision-recall curves, and confusion matrices.  
5. **Extension:** Expanded model to detect or classify other respiratory diseases such as bronchitis, pneumonia, and asthma.

## 📊 Results
| Model | Accuracy | F1-Score | Notes |
|-------|-----------|----------|-------|
| Logistic Regression | 86% | 0.84 | Baseline |
| Random Forest | 91% | 0.90 | Strong feature separation |
| CNN | 93% | 0.92 | Used for advanced respiratory dataset |

## 📁 Repository Structure

