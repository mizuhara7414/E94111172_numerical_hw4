import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f1(x):
    """The function to integrate: sin(x) * x^(-1/4)"""
    return np.sin(x) * x**(-1/4)

def f2(x):
    """The function to integrate: sin(x) * x^(-1)"""
    return np.sin(x) * x**(-1)

def simpson_rule(f, a, b, n):
    """Composite Simpson's rule"""
    if n % 2 != 0:
        n += 1  # Make sure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

def improper_integral_substitution(f, a, b, n, p):
    """
    Compute improper integral using substitution
    For integrals of the form ∫_a^b f(x)/(x-a)^p dx, where 0 < p < 1
    """
    # For the first integral: ∫_0^1 sin(x) * x^(-1/4) dx
    # Let t = x^(1-p), then x = t^(1/(1-p))
    # dx = (1/(1-p)) * t^(p/(1-p)) dt
    # The integral becomes ∫_0^1 sin(t^(1/(1-p))) * t^(-p/(1-p)) * (1/(1-p)) * t^(p/(1-p)) dt
    # = (1/(1-p)) * ∫_0^1 sin(t^(1/(1-p))) dt
    
    if p == 1/4:  # For first integral
        def g(t):
            return np.sin(t**(1/(1-p))) * (1/(1-p))
        
        # New bounds after substitution
        a_new = 0
        b_new = 1
        
    elif p == 1:  # For second integral
        # For integrals of the form ∫_a^b f(x)/(x-a) dx
        # Let t = ln(x-a), then x = e^t + a
        # dx = e^t dt
        # The integral becomes ∫_-inf^ln(b-a) f(e^t + a) dt
        
        def g(t):
            return np.sin(np.exp(t)) * np.exp(t) * np.exp(-t)  # sin(e^t)
        
        # New bounds after substitution
        a_new = np.log(a)
        b_new = np.log(b)
    
    else:
        raise ValueError("p must be 1/4 or 1")
    
    # Apply Simpson's rule to the transformed integral
    return simpson_rule(g, a_new, b_new, n)

# Parameters
n = 4  # Number of subintervals

# Calculate first improper integral
result1 = improper_integral_substitution(f1, 0, 1, n, 1/4)

# Calculate second improper integral
result2 = improper_integral_substitution(f2, 1, 4, n, 1)

# Calculate exact values using scipy
exact_result1, _ = integrate.quad(f1, 0, 1)
exact_result2, _ = integrate.quad(f2, 1, 4)

print(f"Problem 4: ")
print(f"a. ∫_0^1 sin(x) * x^(-1/4) dx = {result1:.10f}")
print(f"   Exact value: {exact_result1:.10f}")
print(f"   Error: {abs(result1 - exact_result1):.10f}")
print(f"b. ∫_1^4 sin(x) * x^(-1) dx = {result2:.10f}")
print(f"   Exact value: {exact_result2:.10f}")
print(f"   Error: {abs(result2 - exact_result2):.10f}")