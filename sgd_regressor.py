# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn-darkgrid')

# Generate our target and features using the make_regression function
X, y = make_regression(n_samples=100, n_features=25)

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate model
model = SGDRegressor()

# Train the model
model.fit(X_train, y_train)

# Evaluate how well our model fits to the training set
score = model.score(X_train, y_train)
print(f"R-Squared: {score}")

# Predict on testing set
y_pred = model.predict(X_test)

# Calculate error and print results
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Print intercept and coefficient
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Plot results
x = range(len(y_test))
plt.plot(x, y_test, linewidth=1, label="Actual")
plt.plot(x, y_pred, linewidth=1, label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend(loc='best', shadow=True)
plt.savefig('Images/sgd_regressor.png')
plt.show()
