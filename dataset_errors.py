# RSS, MSE, R2 score Calculations 

import numpy as np
from sklearn.linear_model import LinearRegression

x = [[6], [8], [10], [14],[18]] # diameter (inches)
y = [[7], [9], [13], [17.5], [18]] # price

model = LinearRegression()
model.fit(x,y) 

print("Residual Sum of Squares Calculation!!!!")
print("Residual Sum of Squares:%.4f"%(((model.predict(x))-y)**2).sum())
print()

print("Mean Squared Error Calculation!!!!")
RSS = (((model.predict(x))-y)**2).sum()
l = len(x)
MSE = RSS/l
print("Mean Squared Error:%.4f"%MSE)
print()

# R2 Square check performance when new data arrives
# new test set data
xtest = [[8],[9],[11],[16],[12]]
ytest = [[11],[8.5],[15],[18],[11]]

model1 = LinearRegression()
model1.fit(xtest,ytest) 

xbar = np.mean(xtest) # mean calculation
ybar = np.mean(ytest)
print()

print("Total Sum of Squares ")
# SSmean = sum(x-ybar)
SStot = np.sum((ytest-ybar)**2)  # SSmean/SStot/TSS
print("SSmean:",SStot)

# Residual SSres/Rss X& y fit to train model (From prediction we find residual)
# we compare output. o/p= (y-ytest); ytest are actual values or labels(just for comparision only)
# from prediction & actual value we get residual value . From xtest value we can predict y_predictz

SSres = np.sum((ytest - model.predict(xtest))**2) # actual - predicted [from xtest]
print("Sum of Residuals:",SSres)
print()

Rsquared = 1-(SSres/SStot)
print('R2:',Rsquared)
print()

# using sklearn implementation
print("Rsquared error=%.4f"%model.score(xtest,ytest))
print()

# or method
from sklearn.metrics import r2_score
print("R2 score:",r2_score(ytest,model1.predict(xtest)))