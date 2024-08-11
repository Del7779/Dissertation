#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:38:54 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
k0 = 0.4
k1 = 0.01
k2 = 1
k3 = 1
k4 = 0.2
J3 = 0.05
J4 = 0.05

# function G
def G(u, v, J, K):
    numerator = 2 * u * K
    denominator = (v - u + v * J + u * K + np.sqrt((v - u + v * J + u * K)**2 - 4 * (v - u) * u * K))
    return numerator / denominator


# Euler method to solve ODE (get the steady state ss)
def euler_method_ss(S_values, y0, dt, steps):
    R_values = []
    for S in S_values:
        R = y0
        for _ in range(steps):
            dRdt = k0 * G(k3 * R, k4, J3, J4) + k1 * S - k2 * R  # X=R assumption as X is not needed for this plot
            R = R + dt * dRdt
        R_values.append(R)
    return R_values

# Values of S for bifurcation plot
S_values = np.linspace(0, 15, 100)

# Initial conditions
initial_conditions = [0.1, 0.5]  

dt = 0.1
steps = 10000

all_R_values = []

for y0 in initial_conditions:
    R_values = euler_method_ss(S_values, y0, dt, steps)
    all_R_values.append(R_values)

# Plot
plt.figure(figsize=(8, 6))
for R_values in all_R_values:
    plt.plot(S_values, R_values, '.')

plt.xlabel('Signal (S)')
plt.ylabel('Response (R)')
plt.title('Mutual activation using Euler method')
plt.axvline(x=10, color='k', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()


##
S=6
R = np.linspace(0,1,20)
plt.figure
plt.plot(R,k0 * G(k3 * R, k4, J3, J4) + k1 * S - k2 * R )
