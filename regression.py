import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the parameters for the line equation (y = mx + b)
m = 2.0  # Slope of the line
b = 1.0  # Y-intercept of the line

# Generate random x values
num_points = 100
x = np.random.uniform(0, 10, num_points)  # Adjust the range as needed

# Generate y values with some random noise
noise = np.random.normal(0, 2, num_points)  # Adjust the standard deviation for noise
y = m * x + b + noise

# Plot the data points
plt.scatter(x, y, label="Data Points")
# plt.plot(x, m * x + b, color='red', label="True Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Data Distributed on a Line with Noise")


# Define the linear model (y = mx + b)
def linear_model(params, x):
    m, b = params
    return m * x + b


# Define the objective function to minimize (sum of squared residuals)
def objective(params, x, y):
    return np.sum((y - linear_model(params, x))**2)


# Initial guess for parameters (slope and y-intercept)
params_initial_guess = [1.0, 1.0]

# Use SciPy to minimize the objective function
result = minimize(objective, params_initial_guess, args=(x, y))

# Extract the optimized parameters
m_opt, b_opt = result.x

print(f"Optimal slope (m): {m_opt}")
print(f"Optimal y-intercept (b): {b_opt}")

plt.plot(x, m_opt * x + b_opt, color='red', label="True Line")
plt.show()
