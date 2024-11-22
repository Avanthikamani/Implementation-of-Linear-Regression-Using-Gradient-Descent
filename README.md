# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the requried libray and read tha dataframe
2.write a function computecost to generate the cost function 
3. perform iteration og gradint step with learning rate
4. plot the cost function using gradint decent and generate the requried graph
5. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: M.avanthika
RegisterNumber:  24901279
*/
import numpy as np

from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data= fetch_california_housing()

x= data.data[:,:3]

y=np.column_stack((data.target, data.data[:,6]))

x_train,x_test, y_train,y_test= train_test_split(x,y,test_size= 0.2,random_state=42)

scaler_x= StandardScaler()

scaler_y= StandardScaler()

x_train= scaler_x.fit_transform(x_train)

x_test= scaler_x.fit_transform(x_test)

y_train =scaler_y.fit_transform(y_train)

y_test =scaler_y.fit_transform(y_test)

sgd= SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd= MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train, y_train)

y_pred=multi_output_sgd.predict(x_test)

y_pred=scaler_y.inverse_transform(y_pred)

y_test= scaler_y.inverse_transform(y_test)
print(y_pred)
![Screenshot 2024-11-22 092221](https://github.com/user-attachments/assets/efc0772f-ccae-4d72-b5a5-6a1cc92837f8)
```

## Output:
​![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
