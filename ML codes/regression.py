# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
# Independent variable (e.g., hours studied)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Dependent variable (e.g., exam scores)
y = np.array([2, 4, 5, 4, 5])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values using the trained model
y_pred = model.predict(X)

# Display model parameters
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
