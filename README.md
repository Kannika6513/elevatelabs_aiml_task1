# 🚢 Titanic Dataset - AIML Internship Task

This project performs data cleaning, preprocessing, outlier removal, normalization, and visualization using the Titanic dataset.

---

## 📁 Dataset Overview

The dataset contains passenger information such as age, sex, fare, family details, etc., to predict survival chances.

---

## ✅ Tasks Performed

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



