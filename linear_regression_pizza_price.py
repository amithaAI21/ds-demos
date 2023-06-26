# Pizza Price calculation based on Diameter of the pizza using Linear Regression

# pip install matplotlib

x = [[6], [8], [10], [14],[18]] # diameter (inches)
y = [[7], [9], [13], [17.5], [18]] # price

# importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y) # teaching the model

print("Coefficent/slope of the model:",model.coef_)
print("Intercept/Constant:",model.intercept_)

# 12 inch pizza (new pattern) should be in 2D. 68 is error
print("12 inch pizza cost:$%.4f"%model.predict([[12]]))
print()

print("Diameter calculation based on x inputs")
print(model.predict(x)) 
print()


# creating own data using linspace(create data within a range, here range also included)
print("Data Prediction using new dataset-linspace()!!!")
v1 = np.linspace(0,25,20) # 0 to 25 range,20 number of datas
print(v1.shape) # v1 should be in 2D
val1 = v1.reshape(20,1)
print(val1.shape)
print(model.predict(val1)) # find price by diameter for test data
print()

print("Plotting the Pizza Piza Against Diameter!!!")
print()
import matplotlib.pyplot as plt
plt.title("Pizza Price against Diameter")
plt.xlabel("Diamter in inches")
plt.ylabel("Price in dollars")
plt.grid(True)
plt.axis([0,25,0,25])
plt.scatter(x,y)     #  scatter plot
plt.plot(x,model.predict(x),c='r')
plt.show()
print()

print("Residual Sum of Squares Calculation!!!!")
print("Residual Sum of Squares:%.4f"%(((model.predict(x))-y)**2).sum())
print()

print("Mean Squared Error Calculation!!!!")
RSS = (((model.predict(x))-y)**2).sum()
l = len(x)
MSE = RSS/l
print("Mean Squared Error:%.4f"%MSE)