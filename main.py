# This is the Complete Lifecycle of machine learning Project , how do you work from data acquisition to preprocessing? --> then you train the model, test the model and then you evaluate it.

import pandas as pd
from sklearn.model_selection import train_test_split  # train the data for cleaning and testing part

# import these three algorithms and models for regression ML problem
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# import R2 score matrix
from sklearn.metrics import r2_score

# ------------------- Part for data preparation -----------------------------

# create DataFrame and store my Dataset into that
df = pd.read_csv('insurance.csv')

# This will print what we have in our dataframe.
print(df.head())

# This line will print no. of rows and columns
print(df.shape)

# print datatypes of the column
print(df.info())

# to get the idea about numerical features like mean, std division, min, max
print(df.describe())

# for null values in our dataset
print(df.isnull().sum())

# for categorical feature like data in string(male , female) we can encode data using replace method of panda library
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
print(df.head())
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
print(df.head())
df.replace({'region': {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
print(df.head())

# separate dependent/target variable(y)[here, charges are only dependent] and independent feature(x) features
x = df.drop(columns=['charges'], axis=1)  # axis=0 is for rows and axis=1 for columns
print(x)
y = df['charges']
print(y)

# train the data for cleaning and testing part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 80% data for training and 20% data for testing
print(x_train.shape)
print(x_test.shape)

# --------------------------------------------------------------------------------------

# --------------- Now, let's import a different machine learning model ------------------
# Since this is regression ML problem , so I will import three different ML models in a regression: Linear regression, decision tree & random forest. ---> then I'll compare the outputs and will see which is working the best.
# Here, we will use R2 score as evaluation matrix.

# below steps for train three ML models-------------
# create one instance for Linear Regression
lr = LinearRegression()
print(lr.fit(x_train, y_train))
# create one instance for Decision Tree Regression
dtr = DecisionTreeRegressor()
print(dtr.fit(x_train, y_train))
# create one instance for Random Forest Regression
rfr = RandomForestRegressor()
print(rfr.fit(x_train, y_train))

# Test the Data--------
# store the prediction made by these three different models
lr_pred = lr.predict(x_test)
dtr_pred = dtr.predict(x_test)
rfr_pred = rfr.predict(x_test)

# calculate r2_score for all these three different ML models----------calculate Outcomes of different models.
print("Linear Regression ->", r2_score(y_test, lr_pred))
print("Decision Tree Regression ->", r2_score(y_test, dtr_pred))
print("Random Forest Regression ->", r2_score(y_test, rfr_pred))
# these are the performance of different ML models that I used to solve Regression UseCase. Once we are satisfied with this model performance, we can use this for deployment.

# ------------------------------------------------------------------------------------------------------------
