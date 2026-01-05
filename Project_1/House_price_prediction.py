# Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from numpy.random.mtrand import random

# Generating an Random data 
np.random.seed(42)
house_size = np.random.randint(500, 3500, 100)
prices = house_size * 150 + np.random.randint(-50000, 50000, 100)

# Create DataFrame
df = pd.DataFrame({'size' : house_size,'price' : prices})
missing_index = np.random.choice(df.index, 5, replace  = False)
df.loc[missing_index, 'price'] = np.nan

# Adding Outliers into price
df.loc[98, 'price'] = 50000
df.loc[99, 'price'] = 2000000


df.head()
df.info()

# Exploring the data
print(df.describe())


# Visualizing the data for understanding
plt.figure(figsize=(10, 6))
plt.scatter(df['size'], df['price'], alpha = 0.5)
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Raw data - Befor cleaning')
plt.show()

# Printing null values 
print(f"\n Misssing value: \n{df.isnull().sum()}")


# Handling the missing values 
df['price'].fillna(df['price'].median(), inplace = True)

# Removing Outliers Using IQR Method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print(f"Orginal Data: {len(df)} rows")
print(f"clean data{len(df_clean)} rows")
print(f"Removed{len(df) - len(df_clean)} outliers")


# Traning the Model
X = df_clean[['size']]
Y = df_clean['price']

X_train, X_test, Y_train, y_test  = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Model selectiona and traning 
model = LinearRegression()
model.fit(X_train, Y_train)

print(f"Model trained!")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Make Predictions
y_pred = model.predict(X_test)
comparsion = pd.DataFrame({
    'Actual': y_test.values[:5],
    'predicted' : y_pred[:5],
    'Difference' : y_test.values[:5] - y_pred[:5]
})

print(comparsion)


# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# Visualize Results
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs Predicted')

# Plot 2: Regression line
plt.subplot(1, 2, 2)
plt.scatter(X_train, Y_train, alpha=0.3, label='Training data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.legend()
plt.title('Model Fit')

plt.tight_layout()
plt.show()