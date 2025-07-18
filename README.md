# Loan-Approval-Model

To predict loan approval using machine learning based on applicant attributes such as income, credit history, employment, and education.
The loan approval prediction model is developed through a structured and data-driven approach involving data cleaning, feature engineering,
visualization, and model evaluation.

Data Acquisition.

Source: An Excel file (loan.xlsx) containing 614 loan application records.
Reason: Real-world dataset with common customer information relevant for financial institutions.

Initial Exploration.

Checked for null values and data types.
Identified missing data in columns like Gender, Married, Dependents, etc.
Ensured that all relevant features and the target column (Loan_Status) were included.
Data Preprocessing.

Handling Missing Values

from sklearn.impute import SimpleImputer

Categorical columns (Gender, Married, etc.): Imputed using most frequent strategy.
Numerical columns (LoanAmount, LoanAmount_log): Imputed using mean strategy.
Most frequent strategy is good for categorical data when to find missing data. 
Mean works well for numerical data when the distribution is not heavily skewed.

Feature Engineering.

Created LoanAmount_log and TotalIncome_log to normalize skewed distributions.
Used log transformations to stabilize variance and normalize features for better model performance.
LoanAmount and Income often exhibit right-skewed distributions. Log transformation makes them more Gaussian-like, benefits the ML models.

Feature Selection.

choosing the most relevant variables from a dataset that contribute significantly to the prediction of the target variable. 
In this case Loan_Status, signifies whether the loan is approved.

Exploratory Data Analysis (EDA)

Used Seaborn countplot to visualize:

Loan approval by Gender,Marital status,Dependents,Self-employment and Credit history vs Loan status.

Helps understand patterns, correlations, and potential biases in the data.

Train-Test Split.

80% for training, 20% for testing.

By splitting the data, we simulate a real-world scenario where the model sees new data and must make predictions accurately.

Encoding Categorical Variables.

Converted categorical columns into numerical using Label Encoding to provide numeric values for machine learning model.\

Feature Scaling.

To normalize your feature data making all numeric features have a mean= 0 and a standard deviation=1 so the model can perform more accurately and efficiently.

Model Training & Evaluation

from sklearn.ensemble import RandomForestClassifier

Accuracy: 78.05%

Strength: Robust to overfitting, handles both numerical and categorical data well.

Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

Accuracy: 70.73%

Strength: Simple and interpretable but prone to overfitting.

Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

Accuracy: 82.93%

Strength: Very fast, works surprisingly well with feature independence assumption.

Model Evaluation

Used confusion matrix and Seaborn heatmap to visualize the performance.

The loan approval prediction model is developed through a structured and data-driven approach involving data cleaning, feature engineering, visualization, and model evaluation. I have compared multiple models to determine which classifier offers the best trade-off between accuracy, speed, and complexity, 
the Gaussian Naive Bayes provides the highest accuracy of 82.93%, demonstrating strong predictive capability on unseen data. Key features influencing loan approval were found to be Credit History, Total Income, and Loan Term, aligning with real-world lending practices.
This project not only provides a functional machine learning solution but also highlights how thoughtful preprocessing and model selection can significantly impact predictive accuracy. The model is suitable for deployment in financial decision-support systems, with potential.

