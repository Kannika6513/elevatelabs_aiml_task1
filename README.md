# 🚀 AI/ML Internship – **Elevate Labs**  
📂 *Repository for Internship Tasks & Projects*  

---

## 📜 Overview  
This repository contains a collection of **Machine Learning** tasks and projects completed during my internship at **Elevate Labs**.  
Each task focuses on different aspects of **data preprocessing, exploratory data analysis, model building, and evaluation**.  

---

## 📑 Contents  

1. 🚢 **Titanic Dataset – Data Cleaning, Preprocessing & Visualization**  
2. 📊 **Titanic Dataset – Exploratory Data Analysis (EDA)**  
3. 🏠 **House Price Prediction – Linear Regression**  
4. 🩺 **Breast Cancer Classification – Logistic Regression**  
5. ❤️ **Heart Disease Prediction – Decision Tree & Random Forest**  

---

## 🚢 **Task 1 – Titanic Dataset (Data Cleaning & Preprocessing)**  

**📁 Dataset:** Passenger details (age, gender, fare, family size, etc.) to predict survival chances.  

**🔧 Steps Performed:**  
- 🗂 **Data Loading** – Read dataset using `pandas`.  
- 🩹 **Handling Missing Values:**  
  - Filled `Age` with median.  
  - Filled `Embarked` with mode.  
  - Filled `Cabin` with `"Unknown"`.  
  - Filled `Fare` with median.  
- 🗑 **Dropped Unnecessary Columns:** `Name`, `Ticket`.  
- 🔤 **Encoding Categorical Variables:** Applied one-hot encoding to `Sex` & `Embarked`.  
- 📉 **Outlier Detection & Removal:** Used IQR method on `Age`, `Fare`, `SibSp`, `Parch`.  
- 📏 **Feature Scaling:** Standardized numeric features using `StandardScaler`.  
- 📊 **Visualization:** Boxplots before & after outlier removal.  

**📦 Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`  

---

## 📊 **Task 2 – Titanic Dataset (Exploratory Data Analysis)**  

**🎯 Objective:** Explore patterns & relationships in the Titanic dataset.  

**📌 Highlights:**  
- 📈 **Summary Statistics:** Mean, median, standard deviation, etc.  
- 📊 **Visual Analysis:**  
  - Histograms & boxplots for `Age`, `Fare`  
  - Bar plots for `Pclass` & `Sex` vs `Survived`  
  - Correlation matrix & pairplot for feature relationships  

**🧠 Key Inferences:**  
- 👩 Women had a higher survival rate.  
- 🛳 First-class passengers had better survival chances.  
- 💰 Fare distribution was right-skewed with outliers.  
- 📉 Negative correlation between `Pclass` and `Fare`.  

---

## 🏠 **Task 3 – House Price Prediction (Linear Regression)**  

**📁 Dataset:** `Housing.csv` – 545 records, 13 features.  

**📌 Steps:**  
1. 📥 Import & preprocess data (encode categorical variables).  
2. ✂ Split into train/test sets.  
3. 🏗 Train a `LinearRegression` model.  
4. 📏 Evaluate using MAE, MSE, R² score.  
5. 📊 Plot predicted vs actual prices.  

**📊 Sample Results:**  
- MAE: ~₹9.7 Lakhs  
- R² Score: ~0.65  
- Moderate fit with interpretable coefficients.  

---

## 🩺 **Task 4 – Breast Cancer Classification (Logistic Regression)**  

**📌 Steps:**  
- 🗑 Drop unnecessary columns: `id`, `Unnamed: 32`  
- 🔤 Encode target (`M` → 1, `B` → 0)  
- ✂ Train/test split (80/20) & standardize features  
- 🏗 Train `LogisticRegression` model  
- 📊 Evaluate with confusion matrix, precision, recall, ROC-AUC  
- ⚡ Threshold tuning example at 0.3  
- 📈 Sigmoid curve plot & explanation  

---

## ❤️ **Task 5 – Heart Disease Prediction (Decision Tree & Random Forest)**  

**📌 Project Overview:**  
Uses `heart.csv` to predict heart disease. Models used:  
- 🌳 Decision Tree Classifier  
- 🌲 Random Forest Classifier  

**🛠 Steps:**  
1. Train & visualize Decision Tree (controlled depth to avoid overfitting).  
2. Analyze overfitting by adjusting max depth.  
3. Train Random Forest & compare accuracy.  
4. Interpret feature importances.  
5. Evaluate both models using cross-validation.  

---
# 🌸 TASK 6 KNN Classification on Iris Dataset 🌼

## 🌺 Steps Performed
1. **Dataset**  
   - Used `Iris.csv` dataset.  
   - Dropped ID column.  
   - Encoded species labels into numbers.  
   - Normalized features using `StandardScaler`.  

2. **🌻 Model**  
   - Applied `KNeighborsClassifier` from `sklearn`.  
   - Tested different `K` values to find the best accuracy.  

3. **🌹 Evaluation**  
   - Computed accuracy score.  
   - Created confusion matrix.  

4. **🌷 Visualization**  
   - Plotted decision boundaries using first two normalized features.  
   - Colors: ❤️ (Setosa), 💚 (Versicolor), 💙 (Virginica).  

## 🌼 Output
- Best K value  
- Accuracy score  
- Confusion matrix plot  
- Decision boundary plot

# 🎗 Breast Cancer Classification

## 📌 Task
Train a machine learning model on the **Breast Cancer Wisconsin dataset** 🩺 and visualize **decision boundaries** 🌈 for two selected features.

## 📂 Dataset
- Source: `sklearn.datasets.load_breast_cancer()`
- Target: Malignant (0) / Benign (1)

## 🛠 Steps
1️⃣ Import libraries  
2️⃣ Load & inspect dataset  
3️⃣ Select 2 features for plotting  
4️⃣ Split into train/test sets  
5️⃣ Train classifier (e.g., Logistic Regression, SVM, KNN)  
6️⃣ Plot decision boundaries & accuracy score  

## 🚀 Output
- Graph with decision boundary separation 🖌  
- Accuracy printed in console 📈
