# Data Science Mini Projects Summary

**Author:** Abdul Samad  
**Email:** abdulsamad57738@gmail.com \
**GitHub Repository:** [github.com/abdulsamad577/Internship_tasks](https://github.com/abdulsamad577/Internship_tasks)  

---

This repository contains the solutions for three foundational data science tasks focused on data exploration, classification modeling, and predictive analytics. The objective of these tasks was to enhance practical skills in data handling, visualization, and model building using Python libraries like `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

---
 


This repository contains the solutions for three foundational data science tasks focused on data exploration, classification modeling, and predictive analytics. The objective of these tasks was to enhance practical skills in data handling, visualization, and model building using Python libraries like `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

---

## 📊 Task 1: Exploring and Visualizing the Iris Dataset

**Objective:**  
Understand how to read, summarize, and visualize a simple dataset.

**Approach:**  
- Loaded the Iris dataset using `pandas`.
- Displayed basic dataset information using `.shape`, `.columns`, and `.head()`.
- Created visualizations using:
  - **Scatter Plot** to show relationships between petal and sepal dimensions.
  - **Histogram** to examine the distribution of flower measurements.
  - **Box Plot** to detect outliers and understand value spread.

**Results and Insights:**  
- Found clear separability between different iris species based on petal length and width.
- Histogram showed petal length and width have distinct clusters for each species.
- Box plots revealed some outliers, particularly in petal dimensions.

---

## 💳 Task 2: Credit Risk Prediction

**Objective:**  
Predict whether a loan applicant is likely to default on a loan.

**Approach:**  
- Handled missing values using appropriate techniques (e.g., imputation).
- Visualized key features such as `LoanAmount`, `ApplicantIncome`, and `Education`.
- Trained two classification models: **Logistic Regression** and **Decision Tree**.
- Evaluated models using **Accuracy** and **Confusion Matrix**.

**Results and Insights:**  
- The Decision Tree performed slightly better in terms of classification accuracy.
- Education and LoanAmount were significant factors influencing loan approval.
- Confusion matrix helped understand the trade-off between false positives and false negatives.

---

## 🏦 Task 3: Customer Churn Prediction

**Objective:**  
Identify customers likely to leave the bank.

**Approach:**  
- Cleaned the dataset and handled missing values.
- Encoded categorical variables like `Geography` and `Gender` using Label Encoding and One-Hot Encoding.
- Trained a **XGBoost Classifier Algorithm**.
- Analyzed **feature importance** to understand key churn drivers.

**Results and Insights:**  
- Achieved **accuracy of 87%**, **precision of 76%**, but a lower **recall of 49%**.
- Key features influencing churn included **Credit Score**, **Age**, and **Balance**.
- Feature importance helped in understanding what factors need attention for churn prevention.

---

## ✅ Tools & Libraries Used
- Python (Pandas, NumPy)
- Matplotlib, plotly & Seaborn (Data Visualization)
- Scikit-Learn (Machine Learning)
- Jupyter Notebook

---

## 📌 Conclusion

These mini projects provided hands-on experience in data wrangling, visualization, machine learning model building, and result interpretation. They serve as a foundational step toward more advanced data science applications.
