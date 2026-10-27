# CSE422 Bank Marketing Classification Project

This project analyzes a Portuguese bank's direct-marketing campaign data and predicts whether a client will subscribe to a term deposit.

The project is implemented in a Jupyter Notebook and includes exploratory data analysis, data preprocessing, supervised classification, unsupervised clustering, and model comparison.

## Project Objective

The target variable is `y`:

- `yes`: the client subscribed to a term deposit
- `no`: the client did not subscribe

This is a supervised binary-classification problem. The project also uses K-Means clustering to identify groups of clients without using the target label.

## Repository Contents

| File | Description |
|---|---|
| `CSE422_Bank_Marketing_Project.ipynb` | Main notebook containing the complete analysis and machine-learning workflow |
| `Bank Marketing.csv` | Bank marketing dataset with client, campaign, and economic features |

## Dataset

The dataset contains 41,188 records and 20 input features. The features describe:

- Client information: age, job, marital status, education, and default status
- Existing products: housing loan and personal loan
- Campaign information: contact method, month, day of week, call duration, and campaign count
- Previous-contact information: days since the previous contact, previous contacts, and previous campaign outcome
- Economic indicators: employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, and number of employees

The target column is `y`.

The target classes are imbalanced: approximately 88.7% of clients did not subscribe and 11.3% did subscribe. Therefore, accuracy is evaluated together with precision, recall, and ROC-AUC.

## Methods

### Exploratory Data Analysis

The notebook includes:

- Dataset shape and feature-type inspection
- Class-distribution analysis
- Correlation heatmap
- Histograms of numerical features
- Bar charts for important categorical features
- Call-duration analysis by subscription outcome
- Monthly campaign-outcome analysis

### Preprocessing

The preprocessing workflow:

1. Encodes categorical columns using `LabelEncoder`
2. Checks for missing values
3. Separates the target column from the input features
4. Standardizes features using `StandardScaler`
5. Splits the data into 80% training and 20% testing sets
6. Uses stratification to preserve the class distribution

### Supervised Learning Models

The following models are trained and compared:

- K-Nearest Neighbors (`K=7`)
- Decision Tree (`max_depth=8`)
- Logistic Regression
- Gaussian Naive Bayes
- Neural Network / MLP (`128-64` hidden-layer architecture)

### Unsupervised Learning

K-Means clustering is applied to the scaled feature set:

- The Elbow Method is used to inspect possible cluster counts
- A final model with `K=3` is trained
- PCA reduces the data to two dimensions for visualization

### Evaluation Metrics

Each supervised model is evaluated using:

- Accuracy
- Precision
- Recall
- Confusion matrix
- ROC curve
- ROC-AUC score

## Results Summary

According to the notebook analysis:

- The Neural Network achieves the best overall performance, with approximately 92.15% accuracy and 0.9494 ROC-AUC.
- Decision Tree and Logistic Regression provide competitive performance with better interpretability.
- Naive Bayes obtains the highest recall among the compared models, which may be useful when identifying as many potential subscribers as possible is important.
- KNN performs reasonably but is more affected by class imbalance and the high-dimensional feature space.

The feature `duration` is strongly associated with the target, but it is only known after a call ends. This should be considered when using the model for real-time campaign targeting.

## Requirements

Install Python 3.9 or later and the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Clone or download this project.
2. Place the dataset and notebook in the same folder.
3. Open a terminal in the project folder.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `CSE422_Bank_Marketing_Project.ipynb`.
6. Run the cells from top to bottom.

### Dataset Filename Note

The notebook currently loads the dataset using:

```python
pd.read_csv('Bank_Marketing.csv')
```

The dataset included in this project is named `Bank Marketing.csv`. Before running the notebook, either rename the CSV file to `Bank_Marketing.csv` or update the `pd.read_csv(...)` line to:

```python
pd.read_csv('Bank Marketing.csv')
```

## Limitations

- The dataset is highly imbalanced, so accuracy alone can be misleading.
- Label encoding categorical variables imposes numeric ordering that may not represent a true relationship between categories.
- Economic variables are highly correlated with one another.
- The `duration` feature may cause data leakage for some real-world deployment scenarios because it is available only after the call.
- K-Means is unsupervised, so its clusters do not necessarily correspond directly to the subscription labels.

## Academic Context

This project was developed for CSE422 and demonstrates practical applications of:

- Exploratory data analysis
- Feature preprocessing
- Supervised machine learning
- Unsupervised machine learning
- Model evaluation and comparison
