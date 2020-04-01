#here i am including the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#including the data set file
#dataset is saved in housing.csv file
df = pd.read_csv('housing.csv')
#print(df.head())
df.head() 

size =df['lotsize']
amount=df['price']
#plotsize(x) is independent variable and price(y) is dependent variable
x = np.array(size).reshape(-1,1)
y = np.array(amount)

#initial plot of dataset
plt.scatter(x,y)
#x axis is labeled as PlotSize
plt.xlabel('PlotSize')
#y axis is labeled as Price
plt.ylabel('Price')

#using linear regression model to train the dataset
from sklearn.model_selection import train_test_split
#30% data is for testing and rest is for training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
pred = linear.predict(x_test)

#plot for training dataset
plt.scatter(x_train,y_train,color='r')
plt.plot(x_train, linear.predict(x_train), color = 'g')
plt.title("Plot for Train DataSet")
plt.xlabel("PlotSize")
plt.ylabel("Price")
#plt.show()

#plot for testing dataset
plt.scatter(x_test,y_test,color='r')
plt.plot(x_train, linear.predict(x_train), color = 'blue')
plt.title("Plot for Test DataSet")
plt.xlabel("PlotSize")
plt.ylabel("Price")
