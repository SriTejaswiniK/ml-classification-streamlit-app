# ML Classification Models Application

# Auto Insurance Fraud Detection using ML

## Problem Statement

Insurance fraud detection is an important problem in the insurance industry. Fraudulent claims lead to significant financial losses. The goal of this application is to build and compare multiple ML classification models to accurately detect fraudulent auto insurance claims.

---

## Dataset Description

- **Dataset Name:** Auto Insurance Claims Fraud Dataset
- **Source:** https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
- **Target Variable:** `fraud_reported` (Y/N)
- **Number of Instances:** 1000
- **Number of Features:** 38

The dataset used in this aaplication is the Auto Insurance Claims Fraud Dataset from Kaggle. It includes data related to insurance policies, claim incidents, customer demographics, nature and severity of the accident etc. It contains a mix contains a mix of numerical and categorical features. Since fraud detection datasets are typically imbalanced, multiple evaluation metrics like prescion, recall, f1 score etc are used aprt from accuracy.

Fraud detection datasets are typically imbalanced, with less number of fraudulent cases compared to legitimate claims. In this dataset Class 0 is Non-Fraudulent Claims and Class 1 is Fraudulent Claims.Because of this imbalance, relying solely on accuracy is misleading. Therefore, metrics such as Recall, AUC, F1-score, and MCC are more meaningful for evaluating performance.

In fraud detection problems, Recall is particularly important, as missing a fraudulent claim (false negative) can result in financial loss.

---

## Models used

The following six ML classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## Evaluation Metrics Used

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Comparison Table

| ML Model            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.840    | 0.856 | 0.635     | 0.816  | 0.714    | 0.615 |
| Decision Tree       | 0.795    | 0.626 | 0.583     | 0.571  | 0.577    | 0.442 |
| KNN                 | 0.750    | 0.623 | 0.480     | 0.245  | 0.324    | 0.207 |
| Naive Bayes         | 0.660    | 0.703 | 0.380     | 0.612  | 0.469    | 0.253 |
| Random Forest       | 0.790    | 0.857 | 0.606     | 0.408  | 0.488    | 0.373 |
| XGBoost             | 0.795    | 0.844 | 0.595     | 0.510  | 0.549    | 0.420 |

---

## Model Performance Observations

## Model Performance Observations

| ML Model Name       | Observation about model performance                                                                                                                                                                                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression | Achieved the best overall performance with high recall (0.816), strong AUC (0.856), and the highest MCC (0.615). Effectively identifies fraudulent cases while maintaining good balance and discrimination. Metrics suggest approximately linear decision boundaries after preprocessing and encoding. High recall indicates minimized false negatives, critical in fraud detection. |
| Decision Tree       | Showed moderate performance but lower AUC (0.626), indicating weaker separation capability between fraud and non-fraud classes. Metrics suggest possible overfitting to local patterns in training data while failing to generalize.                                                                                                                                                 |
| K-Nearest Neighbors | Performed poorly in detecting fraud cases with very low recall (0.245). Struggled with high-dimensional features after encoding. Fraudulent samples were not grouped effectively with similar neighbors, leading to many missed fraud cases.                                                                                                                                         |
| Naive Bayes         | Achieved moderate recall (0.612) but lower precision, resulting in more false positives. Likely due to Naive Bayes’ assumption of feature independence, while some features in the dataset are correlated.                                                                                                                                                                           |
| Random Forest       | Achieved the highest AUC (0.857), showing strong classification capability. Recall was moderate (0.408), meaning some fraudulent cases were missed. Suggests the model leaned toward predicting the majority class (non-fraud), given dataset imbalance.                                                                                                                             |
| XGBoost             | Showed balanced performance with good AUC (0.844) and moderate recall (0.510), but did not outperform Logistic Regression. Gradient boosting captures complex feature interactions, but with only 1000 samples, the added complexity did not translate into superior fraud detection.                                                                                                |

---

## Overall Conclusion

Among all the models used above, Logistic Regression performed best for this dataset due to High Recall (less no. of fraudulent claims missed), High AUC (good class discrimnation), High MCC (best overall balanced performance) and best generalization on small dataset. It achieved highest recall and strong AUC score, making it the most suitable model for fraud detection where minimizing false negatives is critical.

Although ensemble methods like Random Forest and XGBoost performed competitively, they did not surpass Logistic Regression in identifying fraudulent cases, which we can obeserve through their metrics like Recall and MCC.
Hence we can say that for datasets with moderate size and well engineered features, simpler linear models like logistic regression can outperform more complex ensemble methods liek random forest and xgboost.
