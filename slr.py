import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#split datastet into train and the test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=  train_test_split(x,y,test_size=1/3, random_state=0)

#fit thesimple linear regresion into train set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

#predict test set result
y_pred = regressor.predict(x_test)

#visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('ssalary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualizing test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()