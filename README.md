# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
date: 5/10/23
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.


## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Bala Umesh 
RegisterNumber:  212221040024
``` py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
### data.head():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/3a08be70-3b21-4729-8c4c-0be5a731c378)


### data.info():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/a956fb44-7c3d-4e3d-a69c-b0a1cfc96e2d)


### isnull() and sum():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/f6d7fdea-3bbf-476d-b01a-7052083ee593)


### data.head() for position:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/0c416a90-0910-45a6-b097-f8cbfc14a5eb)

### mse value:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/036e4c98-80f8-40be-9a67-e9d33a1f61d9)


### r2 value:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/fef4941c-70fa-4d1c-bbae-080b055b8884)

### prediction value:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113031742/6196ad80-33ae-42c1-afa6-4279f9ce0cc4)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
