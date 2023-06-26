# MANUAL SLOPE_INTERCEPT CALCULATION FOR PIZA PRICE

# import necessary libraries
import numpy as np
from itertools import chain # flatening

# dataset
x = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

# Variance calculation for x and y
variance_x=np.var(x,ddof=1) # ddof=1->n-1
print("Varaince of X:",variance_x)
variance_y = np.var(y,ddof=1)
print("Variance of Y:",variance_y)
print()

# x1 = list(chain.from_iterable(x))

# Covariance calculation for x and y
covar =  np.cov([6,8,10,14,18],[7,9,13,17.5,18],ddof=1) # covariance should be in 1D
print(covar) # we will have matrix 
# varx    covar 
# covar   vary

print("finding beta/slope")
beta = covar[0][1]/variance_x
print("Slope:",beta)
print()

xbar = np.mean(x)
ybar = np.mean(y)
alpha = ybar - (beta * xbar)
print("Intercept:",alpha)

y = 1.9655172413793114 + 0.9762931034482758*12 # 12 inch 
print("Predicted Prize of 12 inch Priza:",y)      