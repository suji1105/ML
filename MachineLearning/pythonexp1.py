import pandas as pd
import math
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  

np.random.seed(42)
X = np.random.rand(100, 1) * 10  
y = 2 * X + 1 + np.random.randn(100, 1) * 2  
data = pd.DataFrame({'Size': X.flatten(), 'Price': y.flatten()})

average_size_sq = math.sqrt(data['Size'].mean())
print(f"Square root of the average size: {average_size_sq:.2f}")

correlation, _ = stats.pearsonr(data['Size'], data['Price'])
print(f"Pearson correlation between Size and Price: {correlation:.3f}")

X = data[['Size']].values
y = data['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse:.2f}")
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Prediction')
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Linear Regression Model Prediction")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
data['Price'].hist(bins=20, edgecolor='black')
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Prices")
plt.grid(axis='y', alpha=0.75)
plt.show()

print("\nBasic ML application with graphs using pandas, math, numpy, scipy, and scikit-learn completed.")
