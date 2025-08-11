# 🚢 Titanic Dataset - AIML Internship Task

This project performs data cleaning, preprocessing, outlier removal, normalization, and visualization using the Titanic dataset.

---

## 📁 Dataset Overview

The dataset contains passenger information such as age, sex, fare, family details, etc., to predict survival chances.

---

## ✅ Task 1 Performed

### 1. Data Loading
- Loaded the Titanic dataset using `pandas`.

### 2. Handling Missing Values
- Filled missing values in the `Age` column with **median**.
- Filled missing values in the `Embarked` column with **mode** (most frequent value).
- Filled missing values in `Cabin` with "Unknown".
- Imputed `Age` and `Fare` with median values.

### 3. Dropping Unnecessary Columns
- Dropped text columns like `Name` and `Ticket` as they are not useful for modeling.

### 4. Encoding Categorical Variables
- Applied **One-Hot Encoding** to `Sex` and `Embarked` columns using `pd.get_dummies()`.

### 5. Outlier Detection and Removal
- Detected and removed outliers from:
  - `Age`
  - `Fare`
  - `SibSp`
  - `Parch`
- Used the **IQR method** (Interquartile Range) for detection.

### 6. Feature Scaling
- Applied **StandardScaler** to normalize the numerical features (`Age`, `Fare`, `SibSp`, `Parch`) using `sklearn.preprocessing`.

### 7. Visualization
- Created **Boxplots** using `Seaborn` and `Matplotlib`:
  - Before and after outlier removal
  - To compare the distributions of the numerical features

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn.preprocessing`

---
# 📁 Task 2: Exploratory Data Analysis (EDA) – Titanic Dataset
 ---

## ✅ Objective:
To perform exploratory data analysis on the Titanic dataset to understand key features, visualize relationships, detect outliers, and generate basic inferences that can support future machine learning models.

---

## 📊 Task Highlights:

1. **Summary Statistics:**
   - Generated mean, median, standard deviation, and other descriptive stats.
   - Helped understand the distribution and spread of numeric features.

2. **Visual Analysis:**
   - **Histograms & Boxplots** were created for `Age`, `Fare`, and other numeric features to check distribution and outliers.
   - **Bar plots** for categorical features like `Pclass` and `Sex` vs `Survived`.
   - **Correlation Matrix** and **Pairplot** were used to explore relationships between multiple variables.
---

## 🧠 Inference Summary:

- Female passengers had a significantly higher survival rate.
- First-class passengers (higher fare) had better survival odds.
- Fare data was right-skewed and had several outliers.
- Strong negative correlation between `Pclass` and `Fare`.
- Moderate positive correlation observed between `SibSp` and `Parch`.

> 🔍 These insights provide crucial understanding for feature selection and model design in future stages.

---

# 🏠 task 3 House Price Prediction using Linear Regression

This project predicts housing prices using **Linear Regression** on the `Housing.csv` dataset. It includes preprocessing, model training, evaluation, and visualization.

---

## 📁 Dataset
The dataset contains 545 records and 13 features including:
- **Numerical**: area, bedrooms, bathrooms, stories, parking
- **Categorical**: mainroad, guestroom, basement, airconditioning, etc.
- **Target**: `price` (house price)

---

## 📌 Steps Covered
1. **Import & Preprocess** the dataset (encoding categorical variables)
2. **Split** data into train and test sets
3. **Train** a Linear Regression model using `sklearn.linear_model`
4. **Evaluate** model using:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - R² Score
5. **Plot** predicted vs actual prices and print regression coefficients

---

## 📊 Model Results (Sample)
- **MAE**: ~₹9.7 Lakhs  
- **R² Score**: ~0.65  
- **Interpretation**: Moderate fit. Coefficients show influence of each feature on price.

---
# Task 4 Logistic Regression - Breast Cancer Classification

This project demonstrates how to apply **Logistic Regression** for binary classification on the **Breast Cancer Wisconsin dataset**.

## Project Steps
1. **Dataset Selection**
   - Uses `data.csv` (Breast Cancer Wisconsin dataset).
2. **Data Preprocessing**
   - Dropped unnecessary columns: `id`, `Unnamed: 32`
   - Encoded target variable: `M` → 1 (Malignant), `B` → 0 (Benign)
3. **Train/Test Split & Standardization**
   - Used 80% training, 20% testing
   - Standardized features using `StandardScaler`
4. **Model Training**
   - Trained a Logistic Regression model using scikit-learn
5. **Model Evaluation**
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC score
   - ROC Curve plot
6. **Threshold Tuning**
   - Example with threshold = 0.3 to show effect on precision and recall
7. **Sigmoid Function Explanation**
   - Plotted sigmoid curve
   - Explained its role in converting model outputs to probabilities
  
   
# Task 5 Heart Disease Prediction - Decision Tree & Random Forest

## 📌 Project Overview
This project uses the **Heart Disease Dataset** to train and evaluate two machine learning models:
- **Decision Tree Classifier**
- **Random Forest Classifier**

We:
1. Train and visualize a Decision Tree.
2. Analyze overfitting by controlling tree depth.
3. Train a Random Forest and compare accuracy.
4. Interpret feature importances.
5. Evaluate both models using cross-validation.

---

## 📂 Dataset
The dataset used is `heart.csv`, which contains various medical attributes such as:
- Age, sex, chest pain type, blood pressure, cholesterol, etc.
- Target: `1` → Heart disease, `0` → No heart disease.

---







