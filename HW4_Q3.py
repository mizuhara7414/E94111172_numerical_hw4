import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(x, y):
    """The function to integrate: 2y * sin(x) * cos(x)"""
    return 2 * y * np.sin(x) * np.cos(x)

def g1(x):
    """Lower bound function: sin(x)"""
    return np.sin(x)

def g2(x):
    """Upper bound function: cos(x)"""
    return np.cos(x)

def simpson_2d(f, g1, g2, a, b, n, m):
    """
    Composite Simpson's rule for double integral
    ∫_a^b ∫_g1(x)^g2(x) f(x,y) dy dx
    """

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    
    
    result = 0
    
    # For each x value, compute inner integral
    for i in range(n+1):
        if i == 0 or i == n:
            coef_x = 1
        elif i % 2 == 1:  # Odd
            coef_x = 4
        else:  # Even
            coef_x = 2
            
       
        y_lower = g1(x[i])
        y_upper = g2(x[i])
        k = (y_upper - y_lower) / m
        y = np.linspace(y_lower, y_upper, m+1)
        
        # Inner integral using Simpson's rule
        inner_sum = 0
        for j in range(m+1):
            if j == 0 or j == m:
                coef_y = 1
            elif j % 2 == 1:  # Odd
                coef_y = 4
            else:  # Even
                coef_y = 2
                
            inner_sum += coef_y * f(x[i], y[j])
            
        result += coef_x * inner_sum * k / 3
    
    return result * h / 3

def gaussian_2d(f, g1, g2, a, b, n, m):
    """
    Gaussian quadrature for double integral
    ∫_a^b ∫_g1(x)^g2(x) f(x,y) dy dx
    """
    # Gaussian quadrature 
    if n == 3:
        x_i = np.array([-0.774596669241483, 0, 0.774596669241483])
        w_i = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    else:
        raise ValueError("n must be 3")
    
    if m == 3:
        y_i = np.array([-0.774596669241483, 0, 0.774596669241483])
        w_j = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    else:
        raise ValueError("m must be 3")
    
    # Transform from [-1, 1] to [a, b]
    x_transformed = 0.5 * (b - a) * x_i + 0.5 * (b + a)
    
    # Calculate the integral
    result = 0
    for i in range(n):
        x = x_transformed[i]
        y_lower = g1(x)
        y_upper = g2(x)
        
        
        y_transformed = 0.5 * (y_upper - y_lower) * y_i + 0.5 * (y_upper + y_lower)
        
        for j in range(m):
            y = y_transformed[j]
            result += w_i[i] * w_j[j] * f(x, y)
    
    return 0.25 * (b - a) * np.sum([(g2(x) - g1(x)) * w_i[i] for i in range(n)]) * result


a, b = 0, np.pi/4  
n, m = 4, 4 

#  using Simpson's rule
simpson_result = simpson_2d(f, g1, g2, a, b, n, m)

#  using Gaussian quadrature
gauss_result = gaussian_2d(f, g1, g2, a, b, 3, 3)

#  exact value analytically
# For this specific integral, we can compute it directly:
def exact_integral():
    # The exact value would be calculated analytically
    # ∫_0^(π/4) ∫_sin(x)^cos(x) 2y*sin(x)*cos(x) dy dx
    # = ∫_0^(π/4) sin(x)*cos(x) * [y²]_sin(x)^cos(x) dx
    # = ∫_0^(π/4) sin(x)*cos(x) * (cos²(x) - sin²(x)) dx
    # = ∫_0^(π/4) sin(x)*cos(x) * cos(2x) dx
    # = 1/2 * ∫_0^(π/4) sin(x)*cos(x) * (cos²(x) - sin²(x)) dx
    # = 1/2 * ∫_0^(π/4) sin(x)*cos³(x) dx - 1/2 * ∫_0^(π/4) sin³(x)*cos(x) dx
    # Using substitution and integration, the result is 1/8
    return 1/8

analytical_result = exact_integral()

print(f"Problem 3: ")
print(f"a. Simpson's rule (n=4, m=4) result: {simpson_result:.10f}")
print(f"b. Gaussian quadrature (n=3, m=3) result: {gauss_result:.10f}")
print(f"c. Exact value: {analytical_result:.10f}")
print(f"Error (Simpson): {abs(simpson_result - analytical_result):.10f}")
print(f"Error (Gaussian): {abs(gauss_result - analytical_result):.10f}")