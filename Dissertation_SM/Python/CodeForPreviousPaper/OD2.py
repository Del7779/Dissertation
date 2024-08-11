#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:57:57 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model parameters
μm = 0.03   # degradation rate of mRNA
μp = 0.001  # degradation rate of protein
τ = 18.5    # mean delay in minutes
s = 5       # width of the delay distribution, total window is 2s
n = 5       # Hill coefficient
p0 = 100    # normalized repression threshold
segments = 10  # Number of segments to approximate the delay distribution

# Define the system of equations including the distributed delay
def model(t, Y):
    M = Y[0]
    P = Y[1:1 + segments]
    dMdt = μm * (1 / (1 + (P[-1]/p0)**n)) - μm * M
    dPdt = np.zeros(segments)
    dPdt[0] = μp * M - μp * P[0]  # input into the first compartment
    for i in range(1, segments):
        dPdt[i] = (segments/s) * (P[i-1] - P[i])  # transfer between compartments
    return np.concatenate(([dMdt], dPdt))

# Initial conditions: assume some initial conditions for mRNA and protein
initial_conditions = np.zeros(1 + segments)
initial_conditions[0] = 1  # initial mRNA concentration
initial_conditions[1] = 0.2  # initial protein concentration in the first compartment

# Time domain of the simulation
t_span = (0, 200)
t_eval = np.linspace(0, 200, 4000)

# Solving the system
solution = solve_ivp(model, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(solution.t, solution.y[0], label='mRNA (M(t))')
plt.plot(solution.t, np.mean(solution.y[1:], axis=0), label='Mean Protein (P(t))')
plt.title('Dynamics of Hes1 Model with Distributed Time Delay')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration')
plt.legend()
plt.ylim([0,50])
plt.grid(True)
plt.show()
