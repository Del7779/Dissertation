#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Parameters
k1 = 2
k2 = 2
k3 = 1
k4 = 1

# Step function for S
def S(t):
    if t < 4:
        return 1.2
    elif t < 8:
        return 1.4
    elif t < 12:
        return 1.6
    elif t < 16:
        return 1.8
    else:
        return 2

def euler_method(y0, t, dt):
    # y0 is a vector
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        R, X = y[i-1]
        St = S(t[i-1])
        dRdt = k1 * St - k2 * X * R
        dXdt = k3 * St - k4 * X
        y[i] = y[i-1] + dt * np.array([dRdt, dXdt])
    return y

# Initial conditions
R0 = 1
X0 = 1.2
y0 = [R0, X0]

# Time points
dt = 0.001
t = np.arange(0, 20, dt)
sol = euler_method(y0, t, dt)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], 'k', label='R')
plt.plot(t, [S(ti) for ti in t], 'r', label='S')
plt.plot(t, sol[:, 1], 'g', label='X')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Adapted signal-response curve')
plt.grid(True)
plt.show()
