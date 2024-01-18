# Machine Learning Project: Complete Lifecycle for Regression Use Case

## Overview

This repository presents a comprehensive guide to the end-to-end lifecycle of a machine learning project, focusing on solving a regression problem. The project encompasses key stages such as data acquisition, preprocessing, model training, testing, and evaluation.

### Key Features

- **Data Preparation:**
  - Load the dataset and perform exploratory data analysis.
  - Encode categorical features for modeling.

- **Model Training and Evaluation:**
  - Utilize three regression models: Linear Regression, Decision Tree Regression, and Random Forest Regression.
  - Evaluate model performance using the R2 score as the evaluation metric.

- **Model Comparison:**
  - Compare the performance of the three models to identify the most suitable for the regression use case.

### Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/abhipatel35/ML-Regression-Lifecycle.git
   ```

2. **Navigate to the Project Directory:**

3. **Install Dependencies:**

4. **Run the Jupyter Notebook or Python Script:**
   - Open and run the Jupyter Notebook/Pycharm to execute the Python script `main.py` to explore the complete project.

### Project Structure

- `main.py`: Python script with the main project code and Jupiter notebook/ Pycharm code containing the complete project code with explanations.
- `insurance.csv`: Sample dataset for the regression use case.


## Data Preparation

### Loading the Dataset
```python
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('insurance.csv')
```

### Exploratory Data Analysis
```python
# Display the first few rows of the dataset
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display data types of each column
print(df.info())

# Statistical summary of numerical features
print(df.describe())

# Check for null values
print(df.isnull().sum())
```

### Data Encoding for Categorical Features
```python
# Encode categorical features
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
```

### Separating Dependent and Independent Variables
```python
# Separate dependent/target variable (y) and independent features (x)
x = df.drop(columns=['charges'], axis=1)
y = df['charges']
```

### Train-Test Split
```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
```

## Model Training and Evaluation

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

# Create and train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions and evaluate
lr_pred = lr.predict(x_test)
print("Linear Regression ->", r2_score(y_test, lr_pred))
```

### Decision Tree Regression
```python
from sklearn.tree import DecisionTreeRegressor

# Create and train the Decision Tree Regression model
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)

# Make predictions and evaluate
dtr_pred = dtr.predict(x_test)
print("Decision Tree Regression ->", r2_score(y_test, dtr_pred))
```

### Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor

# Create and train the Random Forest Regression model
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)

# Make predictions and evaluate
rfr_pred = rfr.predict(x_test)
print("Random Forest Regression ->", r2_score(y_test, rfr_pred))
```

## Model Comparison
After training and evaluating the three models, you can compare their performance using the R2 score. Choose the model that best suits your use case.

Once satisfied with the model's performance, it can be deployed for real-world applications.
