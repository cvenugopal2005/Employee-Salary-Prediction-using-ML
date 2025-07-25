# 💼 Employee Salary Classification Using Machine Learning

This project was developed as part of the **AICTE–IBM SkillsBuild Internship** under the mentorship of **Edunet Foundation**.

## 📌 Objective

To build and deploy a machine learning model that classifies whether an employee earns **>50K** or **≤50K** per year, based on demographic and employment-related features.

## 📂 Dataset

- **Source**: Provided by internship mentor (modified from UCI Adult Income Dataset)
- **Features Used**:
  - `age`, `workclass`, `fnlwgt`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
- **Target Variable**: Binary classification — income >50K or ≤50K

## ⚙️ Technologies & Libraries Used

- Python
- Jupyter Notebook
- Streamlit
- scikit-learn (`LabelEncoder`, `GradientBoostingClassifier`)
- pandas, numpy
- joblib (for model serialization)

## 🚀 Steps Followed

1. Data cleaning: removed invalid rows and handled missing values (`?`)
2. Label encoding for categorical features
3. Feature selection and train-test split
4. Trained multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, etc.)
5. Saved the best-performing model as `best_model.pkl`
6. Built an interactive `Streamlit` web app to predict salaries using the model

## 📊 Results

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: ~85% (may vary slightly per run)
- Interactive UI accepts both single and batch inputs
- Supports CSV uploads and prediction downloads

## 🖥️ Web App UI

You can run the app locally with:

```bash
streamlit run app.py
