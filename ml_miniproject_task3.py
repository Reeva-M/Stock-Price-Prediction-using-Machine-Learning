# -*- coding: utf-8 -*-
"""ML_miniProject_Task3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1J4cOWnubhjgquTzLJlZKQWnp4YpsFjG4

#Reeva Mishra _ D093 _ D2-1 _ 60009220203

#Loading the dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/GOOG.csv")

df.head()

df.info()

"""#Getting Basic insights by visualizations

Visualizing the opening prices of the data.
"""

plt.figure(figsize=(16,8))
plt.title('Google')
plt.xlabel('Days')
plt.ylabel('Opening Price USD ($)')
plt.plot(df['Open'])
plt.show()

"""Visualizing the high prices of the data."""

plt.figure(figsize=(16,8))
plt.title('Google')
plt.xlabel('Days')
plt.ylabel('High Price USD ($)')
plt.plot(df['High'])
plt.show()

"""Visualizing the low prices of the data.

"""

plt.figure(figsize=(16,8))
plt.title('Google')
plt.xlabel('Days')
plt.ylabel('Low Price USD ($)')
plt.plot(df['Low'])
plt.show()

"""Visualizing the closing prices of the data."""

plt.figure(figsize=(16,8))
plt.title('Apple')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(df['Close'])
plt.show()

df2 = df['Close']

df2.tail()

df2 = pd.DataFrame(df2)

df2.tail()

"""#Prediction 100 days into the future."""

future_days = 100
df2['Prediction'] = df2['Close'].shift(-future_days)

df2.tail()

X = np.array(df2.drop(['Prediction'], axis=1))[:-future_days]
print(X)

y = np.array(df2['Prediction'])[:-future_days]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

"""We have implemented three machine learning models:

---


**Decision Trees**: A non-linear model that partitions the feature space based on decision rules.

---


**Linear Regression**: A linear model that predicts the target variable based on a linear combination of features.


---

**Random Forest:**
 Also known as random decision forests, are an ensemble learning method used for classification, regression, and other tasks.

#Fitting the Model Using Decision Trees
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = df2.drop(['Prediction'], axis=1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future

tree_prediction = tree.predict(x_future)
print(tree_prediction)

lr_prediction = lr.predict(x_future)
print(lr_prediction)

predictions = tree_prediction
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
plt.show()

tree = DecisionTreeRegressor(max_features=5)
tree.fit(x_train, y_train)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree.predict(x_test)))
print("Decision Tree RMSE:", tree_rmse)

"""#Fitting the Model Using Linear Regression"""

import numpy as np   #Linear algera Library
import pandas as pd
import matplotlib.pyplot as plt  #to plot graphs
import seaborn as sns  #to plot graphs
from sklearn.linear_model import LinearRegression   #for linear regression model
sns.set()  #setting seaborn as default
import math

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("/content/GOOG.csv")   #reads the input data
data.head()

x=data[['High','Low','Open','Adj Close','Volume']].values   #input
y=data[['Close']].values

from sklearn.model_selection import train_test_split
#split to train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

lm=LinearRegression()
lm.fit(x_train,y_train)

lm.coef_

lm.score(x_train,y_train)

"""#Run model using Test data"""

predictions = lm.predict(x_test)

"""#Checking R squared value"""

from sklearn.metrics import r2_score
r2_score(y_test, predictions)

"""#Compare the actual and predicted values"""

dframe=pd.DataFrame({'actual':y_test.flatten(),'Predicted':predictions.flatten()})

dframe.head(15)

"""#Plotting Graph"""

graph =dframe.head(10)
graph.plot(kind='bar')
plt.title('Actual vs Predicted')
plt.ylabel('Closing price')

"""#using scatter plot compare the actual and predicted data

"""

fig = plt.figure()
plt.scatter(y_test,predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)
plt.ylabel('Predicted', fontsize=20)

import math
from sklearn import metrics

"""#metrics to find accuracy of continous variables"""

print('Mean Abs value:' ,metrics.mean_absolute_error(y_test,predictions))
print('Mean squared value:',metrics.mean_squared_error(y_test,predictions))
print('root mean squared error value:',math.sqrt(metrics.mean_squared_error(y_test,predictions)))

"""#Performing lasso regression"""

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
lasso_coef = lasso.coef_
lasso_score = lasso.score(x_test, y_test)

print("Lasso coefficients:", lasso_coef)
print("Lasso score:", lasso_score)

"""#Using Random Forest"""

from sklearn.ensemble import RandomForestRegressor

"""# Creating a Random Forest Regressor model"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

"""#Fitting the model to the training data"""

rf_model.fit(x_train, y_train)

"""# Predict on the test data"""

rf_predictions = rf_model.predict(x_test)

"""# Evaluating the model"""

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, rf_predictions))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, rf_predictions))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, rf_predictions)))

"""# Visualizing the actual and predicted values"""

plt.figure(figsize=(16, 8))
plt.title("Actual vs Predicted (Random Forest)")
plt.xlabel("Days")
plt.ylabel("Closing Price USD ($)")
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
plt.show()

"""#Conclusion

Based on the given data and the performance metrics of the different models, the Random Forest model appears to be the best suited for stock price prediction.

- The Random Forest model achieved the lowest Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) values compared to the Decision Tree and Linear Regression models.
- This indicates that the Random Forest model made more accurate predictions on average.

- Therefore, based on this analysis, the Random Forest model is the most suitable for stock price prediction using the provided data.
"""



























