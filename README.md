# Employee Salary Prediction & Churn Analysis (Kaggle-style ML Pipeline)

An end-to-end Machine Learning pipeline developed as part of the **CSAI 253 (Machine Learning)** course at **Zewail City of Science and Technology**. This repository encompasses a dual-phase exploration: a robust regression framework engineered to predict employee salaries and a rigorous classification setup analyzing employee retention/churn patterns.

---

##  Key Project Contributions & Insights

### Phase 2: Salary Regression (Kaggle-Style Competition)
* **Target Variable Realignment:** Discovered through extensive iteration that raw target training yields higher structural integrity over volatile USD conversions, isolating and mitigating major evaluation scale errors (Target Alignment).
* **Dimensionality Reduction:** Built custom semantic aggregation layers (`job_group`) to map highly granular, noisy categorical text inputs into stable, low-cardinality archetypes (*Data Scientist, Data Engineer, Manager, etc.*).
* **Ensemble Learning Framework:** Built and deployed a unified `VotingRegressor` leveraging diversified model architectures to lower inference variance.

### Phase 1: Employee Churn Classification
* **Class Imbalance Engineering:** Tackled severe class asymmetry (9.8% Churn vs. 90.2% Active) through robust metric alignment (F1-Score, Recall) and robust non-linear tree induction.
* **Behavioral Feature Synthesis:** Engineered predictive behavioral indicators such as `Overtime_ratio`, `Projects_per_year`, and `Tenure` to capture non-linear degradation thresholds.

---

##  Model Performance Evaluation

### Phase 2: Salary Prediction (5-Fold CV RMSE)
Tree-based architectures and robust kernel transformations vastly outperformed standard linear benchmarks due to strong non-linear relationships within the feature space.

| Model Hierarchy | 5-Fold CV RMSE |
| :--- | :---: |
| **Random Forest Regressor** | **0.36804** |
| **Voting Regressor Ensemble** | **0.38070** |
| Support Vector Regressor (SVR) | 0.38937 |
| Multi-Layer Perceptron (MLP Neural Network) | 0.40281 |
| Linear Regression (Baseline) | 0.42687 |

### Phase 1: Employee Retention Classifier
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **95.59%** | **100.00%** | **55.36%** | **0.7127** |
| Voting Classifier | 93.10% | 79.17% | 40.99% | 0.5401 |
| Decision Tree | 91.22% | 55.05% | 60.88% | 0.5782 |

---

##  Technology Stack
* **Languages:** Python
* **Core ML Ecosystem:** Scikit-Learn, NumPy, Pandas
* **Visualization:** Seaborn, Matplotlib

##  Team Members (Team 7)
* **Ahmed Mokhtar** *(Exploratory Data Analysis (EDA), KNN, Logistic Regression & Visualizations)*
* **Abdallah Saed** *(Data Ingestion, Target Cleansing, Hyperparameter Optimization)*
* **Merna Ahmed** *(Feature Engineering, Random Forest Design, Ensemble Architectures)*
* **Mohamed Hamdy** *(Decision Trees, Support Vector Implementations)*
