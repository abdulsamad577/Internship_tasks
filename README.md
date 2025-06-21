# Data Science Mini Projects Summary

**Author:** Abdul Samad  
**Email:** abdulsamad57738@gmail.com \
**GitHub Repository:** [github.com/abdulsamad577/Internship_tasks](https://github.com/abdulsamad577/Internship_tasks)  

---

This repository contains the solutions for Four foundational data science tasks focused on data exploration, classification modeling, and predictive analytics. The objective of these tasks was to enhance practical skills in data handling, visualization, and model building using Python libraries like `pandas`, `matplotlib`, `plotly`, `seaborn`, and `scikit-learn`.

---
 


# Task 1: Exploring and Visualizing the Iris Dataset

This project focuses on **exploring** and **visualizing** the famous **Iris dataset** using Python. The Iris dataset is a classic dataset in data science, containing data on three different species of iris flowers, and is commonly used for demonstration in EDA and machine learning.

---

#### Dataset Overview

- **Source:** Seaborn Library (`sns.load_dataset('iris')`)
- **Samples:** 150 rows  
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width  
- **Target:** Species (`setosa`, `versicolor`, `virginica`)

---

#### Objective

- Load and inspect the dataset.
- Explore relationships between different flower features.
- Visualize distributions using:
  - **Scatter Plots**
  - **Histograms**
  - **Box Plots**
- Build foundational skills in **data summarization** and **EDA**.

---

#### Tools & Libraries Used

- Python
- Pandas
- Seaborn
- Matplotlib
- NumPy

---

#### Visualizations Created

- **Scatter Plot:**
  - Petal Length vs Petal Width (color-coded by species)
  - Petal Length vs Sepal Width

- **Histograms:**
  - For each numerical feature (KDE curve added for smooth distribution view)

- **Box Plots:**
  - To identify the spread and detect outliers for all four main features

---

#### Insights & Summary

- **Species Balance:** The dataset is evenly balanced (50 samples per species).
- **Petal Features:** Strong separation between species when visualized via petal length and width.
- **Sepal Features:** Less separation, but still helpful for classification.
- **Outliers:** Some minor outliers in petal width and sepal length.

---

#### Key Skills Practiced

- Dataset inspection using `.shape`, `.columns`, `.info()`, and `.describe()`
- Basic EDA techniques
- Visualization best practices
- Understanding of feature relationships in classification problems

---

# Task 2: Credit Risk Prediction

This project aims to predict whether a **loan applicant** is likely to be approved or not based on their personal and financial profile. Using machine learning models, we perform **credit risk assessment** and provide key insights to assist lenders in decision-making.

---

#### Dataset Overview

- **Source:** [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Train Dataset:** 614 samples  
- **Test Dataset:** 367 samples  
- **Target Variable:** `Loan_Status` (Y = Yes, N = No)

#### Features:
- **Categorical:** Gender, Married, Dependents, Education, Self_Employed, Property_Area
- **Numerical:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- **Identifiers (Dropped):** Loan_ID

---

#### Objective

To build a classification model that can **predict loan approval** and assess the risk of a credit default, using demographic, financial, and employment-related features.

---

#### Tools & Libraries Used

- Python (Pandas, NumPy)
- Visualization: Seaborn, Matplotlib, Plotly
- Machine Learning: Scikit-learn, RandomForest
- Feature Selection: SelectKBest (Chi-squared)
- Data Cleaning: RandomForest-based imputation

---

#### Exploratory Data Analysis (EDA)

- Count plots and histograms to visualize distribution of:
  - **LoanAmount**
  - **ApplicantIncome**
  - **Education vs Loan Status**
- **Box plots** and **scatter plots** to examine income vs. loan relationships
- Visualized **missing values** and their counts

---

#### Data Preprocessing

- Handled missing values using **Random Forest Imputation**
- Applied **Label Encoding** to categorical features
- Selected best features using **Chi-squared Test**

---

#### Model Training & Evaluation

| Model              | Evaluation |
|-------------------|------------|
| Random Forest      | ✅ Best performer |

####  Final Model: **Random Forest Classifier**
- Trained on the selected top features
- Evaluated on hold-out test set
- Generated confusion matrix to assess performance

---

#### Results & Accuracy

- The model demonstrated high accuracy in predicting loan approvals
- The **Random Forest model** proved effective at handling:
  - Missing values
  - Categorical variables
  - Class imbalance (via robust ensemble learning)

---

#### Key Insights

- Applicants with higher **credit history** and **income** are more likely to get approved
- Features such as **LoanAmount** and **Education** influence approval decisions
- Handling missing values properly greatly boosts prediction performance

---

# Task 3: Customer Churn Prediction

This project focuses on predicting which bank customers are likely to **leave the bank (churn)** using classification models. Accurately identifying at-risk customers helps banks improve **retention strategies**, reducing potential revenue loss.

---

#### Dataset Overview

- **Source:** [Kaggle - Predicting Churn for Bank Customers](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)
- **Rows:** 10,000  
- **Columns:** 14 (including target `Exited`)  
- **Target Variable:** `Exited` (1 = Churned, 0 = Stayed)

#### Features:
- **Numerical:** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- **Categorical:** `Geography`, `Gender`
- **Removed:** `RowNumber`, `CustomerId`, `Surname` (non-informative)

---

#### Objective

To build a machine learning model that predicts **customer churn** based on profile and account information. The final model helps in understanding **key churn factors** and provides actionable insights.

---

#### Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn, XGBoost
- ConfusionMatrixDisplay, LabelEncoder

---

#### Models Evaluated

| Model               | Accuracy | F1-Score |
|--------------------|----------|----------|
| Logistic Regression| ~78%     | ~60%     |
| Random Forest      | ~85%     | ~62%     |
| KNN                | ~83%     | ~59%     |
| SVM                | ~84%     | ~61%     |
| **XGBoost**        | **87%**  | **63%**  ✅ *(Best)*

---

#### Final Model: XGBoost Classifier

- **Accuracy:** 87%  
- **Precision:** 72%  
- **Recall:** 55%  
- **F1-Score:** 63%

> The model performs well in identifying most churners, although further improvement in recall would increase its real-world value.

---

#### Key Insights (Top Features)

- **NumOfProducts:** Customers with fewer products tend to churn more.
- **IsActiveMember:** Inactive users are more likely to exit.
- **Age:** Older users show higher churn risk.
- **Geography_Germany:** Customers from Germany churn more.
- **Balance:** Has moderate impact on churn.
- **Gender:** Slight impact, males show slightly more churn.

---

#### Recommendations

- Focus retention campaigns on **inactive, older customers** with few products.
- Pay special attention to **customers from Germany**.
- Consider launching personalized offers to **boost product usage**.

---


# Task 4: Predicting Insurance Claim Amounts

This project aims to predict medical insurance costs using demographic and lifestyle information such as age, BMI, number of children, smoking status, and more. A **Linear Regression model** is applied to the dataset to estimate the insurance charges for individuals.


#### Dataset Information

- **Source:** [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mragpavank/insurance1)
- **Features:**
  - `age`: Age of the individual
  - `sex`: Gender (male/female)
  - `bmi`: Body Mass Index
  - `children`: Number of children covered by insurance
  - `smoker`: Smoker or non-smoker
  - `region`: Area of residence in the US
  - `charges`: Final medical insurance cost (Target Variable)

---

#### Project Objective

- To explore and preprocess the data
- To build a predictive regression model (Linear Regression)
- To evaluate the model using standard metrics like MAE, MSE, RMSE, and R²
- To understand which features most affect insurance charges

---

#### Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

#### Model Performance (Linear Regression)

| Metric | Score |
|--------|--------|
| MAE    | 4198.11 |
| MSE    | 35,901,914.11 |
| RMSE   | 5991.82 |
| R²     | 0.8046 |

> The model explains about **80% of the variance** in the insurance charges.

---

#### Key Insights


- 🚬 **Smokers** are charged significantly more than non-smokers.
- 📈 **Age** and **BMI** show a positive correlation with charges.
- 🧒 The number of **children** has little influence on the total charge.
- 📌 **Smoking status** is the most important factor.

---
## 📌 Conclusion

These mini projects provided hands-on experience in data wrangling, visualization, machine learning model building, and result interpretation. They serve as a foundational step toward more advanced data science applications.
