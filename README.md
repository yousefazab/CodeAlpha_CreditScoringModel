# Credit Scoring Model

### CodeAlpha Machine Learning Internship – Task 1

## Project Overview

This project develops a machine learning classification system to predict customer credit scores based on financial and credit-related information.

Customers are classified into three categories:

* **Poor**
* **Standard**
* **Good**

The project covers the complete machine learning workflow, including data exploration, preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, and model saving.

## Dataset

**Source:** Kaggle – Credit Score Classification Dataset

The dataset contains customer financial information, including income, outstanding debt, credit history, payment behavior, and credit utilization.

| Dataset               | Records |   Features |
| --------------------- | ------: | ---------: |
| Training Dataset      | 100,000 | 28 columns |
| Provided Test Dataset |  50,000 | 27 columns |

For model development, the labeled training dataset was divided into:

* **Training Set:** 80,000 records
* **Testing Set:** 20,000 records

The split was performed without overlapping customers between the two sets.

## Data Preprocessing

The following preprocessing steps were implemented:

1. Removed unnecessary identification columns.
2. Converted numeric features stored as strings into numeric values.
3. Handled invalid and missing values.
4. Applied numerical feature scaling.
5. Encoded categorical features.
6. Created a preprocessing pipeline to ensure consistent transformations.

The final preprocessing pipeline transformed the original input features into **47 model-ready features**, with no remaining missing values.

## Feature Engineering

Additional financial indicators were created to improve model performance:

* Debt-to-Income Ratio
* Loan-to-Income Ratio
* Credit History in Months

These features help the model capture customer financial behavior and credit risk patterns.

## Machine Learning Models

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Initial Model Comparison

| Model               | Accuracy | Weighted F1-Score | ROC-AUC |
| ------------------- | -------: | ----------------: | ------: |
| Logistic Regression |   62.96% |            63.18% |  75.58% |
| Decision Tree       |   65.18% |            65.49% |  75.82% |
| Random Forest       |   69.09% |            69.37% |  82.84% |

Random Forest achieved the strongest initial performance.

## Hyperparameter Optimization

Random Forest was optimized using:

* RandomizedSearchCV
* GroupKFold Cross-Validation
* Weighted F1-Score optimization

Group-based cross-validation was used to prevent customer overlap between training and validation folds.

### Final Model Results

| Metric             | Original Random Forest | Tuned Random Forest |
| ------------------ | ---------------------: | ------------------: |
| Accuracy           |                 69.09% |          **69.79%** |
| Weighted Precision |             **71.52%** |              71.05% |
| Weighted Recall    |                 69.09% |          **69.79%** |
| Weighted F1-Score  |                 69.37% |          **70.06%** |
| ROC-AUC            |                 82.84% |          **83.03%** |

The tuned Random Forest was selected as the final model based on its improved overall weighted F1-score and ROC-AUC.

However, recall for the Poor category decreased from 74.82% to 71.74%, indicating an important trade-off.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* ROC Curves and AUC

### Final ROC-AUC by Class

| Credit Score Category |   AUC |
| --------------------- | ----: |
| Good                  | 0.885 |
| Poor                  | 0.866 |
| Standard              | 0.792 |

## Feature Importance

The five most influential features identified by Random Forest were:

1. Outstanding Debt
2. Interest Rate
3. Delay from Due Date
4. Credit History in Months
5. Debt-to-Income Ratio

Feature importance reflects model reliance on these variables and does not establish causation.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook
* Visual Studio Code

## Project Structure

```text
Task_1_Credit_Scoring_Model/
│
├── Data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── credit_scoring_model.pkl
│
├── credit_scoring_model.ipynb
│
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## How to Run

1. Download or clone the project repository.
2. Install the required Python libraries.
3. Open `credit_scoring_model.ipynb` using Jupyter Notebook or VS Code.
4. Ensure the dataset files are available inside the `Data` folder.
5. Run the notebook cells sequentially to reproduce the analysis and model training.

### Load the Saved Model

```python
import joblib

model = joblib.load(
    "models/credit_scoring_model.pkl"
)

predictions = model.predict(new_customer_data)
```

The prediction input must contain the same raw feature columns used during model training.

## Limitations

* The model achieved approximately 69.79% accuracy on the held-out evaluation set.
* Classification performance differs across credit score categories.
* Customer-level separation was used to reduce data leakage.
* Additional checks for feature availability, fairness, calibration, and real-world generalization are necessary before deployment in financial decision-making.

## Conclusion

This project demonstrates an end-to-end machine learning solution for customer credit score classification.

By combining data preprocessing, financial feature engineering, multiple classification algorithms, and hyperparameter optimization, the final tuned Random Forest achieved **69.79% accuracy and 83.03% ROC-AUC**.

The project was developed as part of the **CodeAlpha Machine Learning Internship – Task 1: Credit Scoring Model**.
