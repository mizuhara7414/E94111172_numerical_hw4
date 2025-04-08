import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    """The function to integrate: ln(x) * x"""
    return np.log(x) * x

def gaussian_quadrature(f, a, b, n):
    """
    Gaussian quadrature for n points
    """
    # Gaussian quadrature points and weights for different n
    if n == 3:
        # Legendre polynomial roots for n=3
        x_i = np.array([-0.774596669241483, 0, 0.774596669241483])
        w_i = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    elif n == 4:
        # Legendre polynomial roots for n=4
        x_i = np.array([-0.861136311594053, -0.339981043584856, 
                        0.339981043584856, 0.861136311594053])
        w_i = np.array([0.347854845137454, 0.652145154862546, 
                        0.652145154862546, 0.347854845137454])
    else:
        raise ValueError("n must be 3 or 4")
    
    # Transform from [-1, 1] to [a, b]
    x_transformed = 0.5 * (b - a) * x_i + 0.5 * (b + a)
    
    # Calculate the integral
    result = 0.5 * (b - a) * np.sum(w_i * f(x_transformed))
    
    return result

# Parameters
a, b = 1, 1.5  # Integration bounds

# Calculate using Gaussian quadrature with n=3 and n=4
gauss_n3 = gaussian_quadrature(f, a, b, 3)
gauss_n4 = gaussian_quadrature(f, a, b, 4)

# Calculate exact value using scipy
exact_result, _ = integrate.quad(f, a, b)

# Calculate exact value analytically
# For ln(x) * x, the indefinite integral is x^2/2 * ln(x) - x^2/4
def exact_integral(x):
    return x**2/2 * np.log(x) - x**2/4

analytical_result = exact_integral(b) - exact_integral(a)

print(f"Problem 2:")
print(f"Gaussian quadrature (n=3) result: {gauss_n3:.10f}")
print(f"Gaussian quadrature (n=4) result: {gauss_n4:.10f}")
print(f"Exact value : {exact_result:.10f}")

