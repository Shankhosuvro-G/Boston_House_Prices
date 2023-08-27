import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

df=pd.read_csv("BOSTON.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe())
print(df.nunique())
df.dropna(inplace=True)
#plt.figure(figsize=(15,15))
#sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
#plt.show()
#plt.scatter(x=df.RM,y=df.AGE,marker='o')
#print(plt.show())
#plt.scatter(x=df.CRIM,y=df.MEDV,marker='*')
#print(plt.show())
x=df[['CRIM','RM','INDUS','TAX','RAD','DIS','LSTAT','NOX','DIS','B','CHAS','PTRATIO','ZN']]
y=df[['MEDV']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=10)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
lr=LinearRegression()
lr.fit(x_train,y_train)
predictions=lr.predict(x_test)
print(predictions)
print("Accuracy Score: ",metrics.r2_score(y_test,predictions))
