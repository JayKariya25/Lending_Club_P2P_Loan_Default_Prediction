# 🏦 **Lending Club P2P Loan Default Prediction**
## **Machine Learning Mini-Project**
This repository contains the complete solution for the Machine Learning Mini-Project focused on predicting loan default risk for a peer-to-peer (P2P) lending platform. The project evaluates five different classifiers to determine the optimal strategy for minimizing financial loss.

### 🎯 **Project Goal :-**
To build and evaluate machine learning models that predict whether a loan applicant will Default (Class 1) or Fully Pay (Class 0), prioritizing Recall to minimize the costly error of missing a default.

### 💾 **Dataset and Tools :-**
This section details the size of the data used for training and lists all the primary software tools (Python libraries) utilized in the project.

- **Dataset Size and Scope :-**
  
The project uses a subset of the complete historical Lending Club loan data, filtered for final loan statuses (```Fully Paid```, ```Charged Off```, or ```Default```).

| Metric | Detail |
|----------|----------|
| Raw Dataset Size | Over **1.3 million** loan records. |
| Working Dataset Size | Approximately **800,000 to 1 million** rows (after filtering to finalized loans and handling missing values). |
| Prediction Focus | Binary Classification (Default vs. Fully Paid). |

- **Python Tools (Libraries) Used :-**

  The following libraries were crucial for data processing, pipeline creation, and model training:

| Category | Primary Libraries | Key Modules/Algorithms Used |
|----------|----------|----------|
| Core ML Models | ```sklearn.ensemble```, ```xgboost``` | Random Forest, XGBoost, Logistic Regression, KNN, MLP. |
| Data Manipulation | ```pandas```, ```numpy``` | DataFrame operations, data cleaning, and numerical array handling. | 
| Preprocessing | ```sklearn.preprocessing```, ```sklearn.compose``` | ```StandardScaler```, ```OneHotEncoder```, ```ColumnTransformer```, ```SimpleImputer```. | 
| Visualization | ```matplotlib```, ```seaborn``` | Generating ROC Curves, Confusion Matrices, and bar chart comparisons. | 
| Workflow | ```sklearn.pipeline``` | Creating the robust machine learning workflow pipeline. | 

### ⚙️ **Setup and Prerequisites :-**

The project is executed using a Jupyter Notebook/Google Colab environment.

- **Software Requirements :-**

Ensure you have Python installed, along with the following libraries:
```
#Core libraries for data analysis and modeling
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

- **Data Preparation (Crucial Step) :-**
   
  The notebook expects the full, original Lending Club dataset due to its rich features required for robust modeling.

  - **Download**: Obtain the large historical Lending Club loan data (e.g., from Kaggle).

  - **Placement**: Place the downloaded CSV file into the same directory as the main notebook.

  - **Naming**: Rename your downloaded file exactly to: ```loan_full_data_kaggle.csv```.

### 🚀 **Execution Guide :-**

- **Open the Notebook**: Launch the main file: ```Loan_Default_Prediction_System.ipynb```.

- **Run Cells**: Execute all cells sequentially. The notebook handles the entire ML workflow:

  - **Data Loading & Filtering** (Only finalized loans are kept).

  - **Preprocessing Pipeline** (Scaling, One-Hot Encoding, Missing Value Imputation).

  - **Model Training** (Logistic Regression, Random Forest, XGBoost, KNN, MLP).

  - **Evaluation & Visualization**.

### 💡 **Key Technical Details :-**
**Target Imbalance**: Handled using explicit ```class_weight='balanced'``` for Logistic Regression and Random Forest, and ```scale_pos_weight``` for XGBoost.

**Data Leakage Prevention**: Over 15 post-loan and irrelevant columns (e.g. ```total_pymnt```, ```last_pymnt_d```, ```id```) were dropped to ensure the model only uses pre-issuance features.

**Performance Metrics Focus**: Evaluation focuses heavily on **Recall (Default)** and **ROC AUC**, as these are superior to simple **Accuracy** in imbalanced financial problems.

### 📈 **Conclusion Highlights :-**

The final model comparison shows a critical trade-off between maximizing general accuracy and minimizing financial risk.

| Model | Highest Metric Achieved | Financial Justification |
|----------|----------|----------|
| Logistic Regression | Recall (Default): 0.8000 | **Best for Risk Mitigation**. It catches 80% of defaulting loans, minimizing the loss of principal (False Negatives). |
| Random Forest | ROC AUC: 0.7200 | **Best for Technical Robustness**. Offers the most stable and highest overall discriminatory power across the entire confidence spectrum. |
| XGBoost | Accuracy: 0.8400 | Highest overall correctness, but its lower Recall makes it riskier for lending. |   

The project concludes that **Logistic Regression** is the most valuable model for a risk-averse investment strategy.

### **Platforms Used :-**

- **Coding Platform:** Google Colab (Jupyter Notebook - Python).

- **Hosting Platform:** GitHub.

### **Dataset Link** : https://www.kaggle.com/datasets/wordsforthewise/lending-club
