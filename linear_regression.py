# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

# Define X and y variables
X = 5 * np.random.rand(100, 1)
y = 4 - 2 * X + np.random.randn(100, 1)

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
model = LinearRegression()

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
def plot_results():
    plt.scatter(X_test, y_test, s=10, color='gray')
    plt.plot(X_test, y_pred, color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Linear Regression')
    plt.savefig('Images/linear_regression.png')
    plt.show()

# Plot Residuals


def plot_residuals():
    plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, color="green", s=10, label='Train data')
    plt.scatter(y_pred, y_pred - y_test, color="blue", s=10, label='Test data')
    plt.hlines(y=0, xmin=-20, xmax=20, linewidth=2)
    plt.legend(loc='upper right')
    plt.title("Residual errors")
    plt.savefig('Images/residual_plot.png')
    plt.show()


plot_residuals()
