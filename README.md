# 🧠 WiDS Datathon 2025 – Brain Connectivity & Socio-Demographic Classification

Participated in the **WiDS (Women in Data Science) Datathon 2025** hosted on Kaggle, focused on predicting **biological sex** and **ADHD diagnosis** using large-scale **fMRI brain connectivity data** and **socio-demographic features**.

> 🏆 Achieved a **macro-F1 score of 0.75458**, ranking **427th out of 1,075** teams (top ~40%).

---

## 🧪 Problem Overview

The goal was to classify:
- **Biological sex** (binary classification)
- **ADHD diagnosis** (binary classification)  
Using:
- Brain functional connectivity matrices (300+ edges)
- Cognitive and demographic variables (age, education, income, etc.)
- Study site and scanner info

---

## 🧰 Tech & Tools

- **Languages/Libraries**: Python, pandas, NumPy, scikit-learn, CatBoost, SHAP  
- **Visualization**: Matplotlib, Seaborn  
- **Platform**: Kaggle Notebooks

---

## 🧠 Approach

- 📊 **Preprocessing**:  
  - Cleaned and merged multi-modal datasets  
  - Applied **Variance Threshold** and **PCA** for dimensionality reduction  
  - Correlation-based feature selection to remove multicollinearity

- ⚙️ **Modeling Pipeline**:  
  - Algorithms: Logistic Regression (ElasticNet), Random Forest, CatBoost  
  - **Stratified K-Fold CV**, stratified by **diagnosis** and **site**  
  - Focused on minimizing LB shakeup with robust validation

- 🔍 **Interpretability**:  
  - Visualized **SHAP values** for brain region interaction importance  
  - Used **confusion matrices** to assess model bias & class performance

---

## 📦 Artifacts

- 📓 **Notebook** (full EDA + ML pipeline):  
  [🔗 View on Kaggle](#) <!-- Replace with actual Kaggle notebook link -->

- 📊 **Leaderboard**:  
  [🔗 View on Kaggle](#) <!-- Replace with actual leaderboard link -->

---

## 📌 Key Highlights

- Combined neuroscience and ML to explore **mental health patterns**
- Ensured **reproducibility** via shared Kaggle Notebook
- Balanced model **accuracy and transparency** using SHAP

---

## 🤝 Acknowledgment

Thanks to Stanford WiDS and Kaggle for the opportunity to contribute to socially impactful AI research.

