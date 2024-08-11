#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:55:46 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
k0 = 0
k1 = 0.05
k2 = 0.1
k2_prime = 0.5
k3 = 1
k4 = 0.2
J3 = 0.05
J4 = 0.05

# function G
def G(u, v, J, K):
    numerator = 2 * u * K
    denominator = (v - u + v * J + u * K + np.sqrt((v - u + v * J + u * K)**2 - 4 * (v - u) * u * K))
    return numerator / denominator

# Function E(R)
def E(R):
    return G(k3, k4*R, J3, J4)

# Euler method
def euler_method(S_values, y0, dt, steps):
    R_values = []
    for S in S_values:
        R = y0
        for _ in range(steps):
            dRdt = k0 + k1 * S - k2 * R - k2_prime * E(R) *R
            R = R + dt * dRdt
        R_values.append(R)
    return R_values

# Values of S for bifurcation plot
S_values = np.linspace(0, 2, 100)

# Initial conditions
initial_conditions = [0.1, 20]  # Different initial conditions to capture multiple branches

dt = 0.01
steps = 20000

all_R_values = []

for y0 in initial_conditions:
    R_values = euler_method(S_values, y0, dt, steps)
    all_R_values.append(R_values)

# Plot
plt.figure(figsize=(8, 6))
for R_values in all_R_values:
    plt.plot(S_values, R_values, '.')

plt.xlabel('Signal (S)')
plt.ylabel('Response (R)')
plt.title('Mutual inhibition')
#plt.axvline(x=0.75, color='k', linestyle='--', label='$S_{crit1}$')
#plt.axvline(x=1.25, color='k', linestyle='--', label='$S_{crit2}$')
plt.legend()
plt.grid(True)
plt.show()



# 
S=10
R = np.linspace(0,1,100)
dRdt = k0 + k1 * S - k2 * E(R) - k2_prime*R
plt.plot(R,dRdt,'.')

