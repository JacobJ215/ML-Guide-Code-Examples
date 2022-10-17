import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# Define target and feature variables
X = 3 * np.random.rand(100, 1)
y = 4 + 2 * X + np.random.randn(100, 1)

iter = 1000  # Number of iterations
lr = 0.01  # Learning Rate

b = np.random.random()  # bias
theta = np.random.randn(2, 1)  # weights


# Plot relation between variables


def plot_reg():
    ax = sns.regplot(x=X, y=y, color="g")
    plt.title("Relationship between x and y")
    plt.savefig("Images/regression_plot.png")
    plt.show()


plot_reg()


# Define function to calculate the cost


def compute_cost(X, y, theta):
    """
    compute cost
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        theta (ndarray (n,)): model parameters
    Returns
        cost (scalar): cost
    """
    m = X.shape[0]
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))

    return cost


# Create function to perform gradient descent algorithm


def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Computes the gradient for linear regression
    Args:
        X (ndarray (m, )): Data, m examples
        y (ndarray (m, )): target values
        theta  (scalar)    : model parameters
        learning_rate (scalar) : linear rate of our model
        iterations (scalr) : number of iterations
    Returns:
        Final theta vector and array of cost history over no of iterations
    """
    m = X.shape[0]

    loss_history = np.zeros(iterations)
    weight_history = np.zeros((iterations, 2))

    for i in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        weight_history[i, :] = theta.T
        loss_history[i] = compute_cost(X, y, theta)

    return loss_history, weight_history, theta


# Instantiate model and create variables for tracking loss history


X_b = np.c_[np.ones((len(X), 1)), X]
cost_history, theta_history, theta = gradient_descent(X_b, y, theta, lr, iter)


# Plot learning curve


def plot_loss():
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set_ylabel('Theta')
    ax.set_xlabel('Iterations')
    ax.plot(range(iter), cost_history, 'b.')
    plt.savefig('Images/gd_learning_curve.png')
    plt.show()


plot_loss()
