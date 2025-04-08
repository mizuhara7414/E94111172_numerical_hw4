import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    """The function to integrate: e^x * sin(4x)"""
    return np.exp(x) * np.sin(4*x)

def trapezoidal_rule(f, a, b, n):
    """Composite trapezoidal rule"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpson_rule(f, a, b, n):
    """Composite Simpson's rule"""
    if n % 2 != 0:
        n += 1  # Make sure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

def midpoint_rule(f, a, b, n):
    """Composite midpoint rule"""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    y = f(x)
    return h * np.sum(y)

# Parameters
a, b = 1, 2  # Integration bounds
n = 10  # Number of subintervals for h = 0.1

# Calculate using all three methods
trap_result = trapezoidal_rule(f, a, b, n)
simpson_result = simpson_rule(f, a, b, n)
midpoint_result = midpoint_rule(f, a, b, n)

# Calculate exact value using scipy
exact_result, _ = integrate.quad(f, a, b)

print(f"Problem 1: ")
print(f"a. Trapezoidal rule result: {trap_result:.10f}")
print(f"b. Simpson's rule result: {simpson_result:.10f}")
print(f"c. Midpoint rule result: {midpoint_result:.10f}")
print(f"Exact value: {exact_result:.10f}")