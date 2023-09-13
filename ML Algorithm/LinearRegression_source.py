!pip install scikit-learn

# block 1
from sklearn.linear_model import LinearRegression 
# Import the linear regression model
import matplotlib.pyplot as plt 
# The plotting library
import numpy as np

# block 2
x = np.array([121, 125, 131, 141, 152, 161]).reshape(-1,1) 
# x denotes the house area as a feature.
y = np.array([300, 350, 425, 405,496,517]) 
# y denotes the house price.
plt.scatter(x,y)
plt.xlabel("area") # X axis indicates the area.
plt.ylabel("price") # Y axis indicates the price.
plt.show()

# block 3 
lr = LinearRegression() # Encapsulate the linear regression model into an object.
lr.fit(x,y) # Train the model on the dataset.

# block 4
w = lr.coef_# Slope of the model
b = lr.intercept_# Intercept of the model
print('Slope: ',w)      # w = slope b = intercept
print('Intercept: ',b)  # y = wx + b

# block 5
plt.scatter(x,y)
plt.xlabel("area") # X axis indicates the area.
plt.ylabel("price") # Y axis indicates the price.
plt.plot([x[0],x[-1]],[x[0]*w+b,x[-1]*w+b])

# block 6
testX = np.array([[130]])# A test sample with an area of 130
lr.predict(testX)





