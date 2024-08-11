#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:06:39 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
'font.family': 'serif',
'font.serif': ['Times New Roman'],
'axes.titleweight': 'bold',  # Bold title
'axes.labelweight': 'bold',  # Bold labels
'font.size': 16,
'axes.titlesize': 18,
'axes.labelsize': 18,
'xtick.labelsize': 14,
'ytick.labelsize': 14,
'legend.fontsize': 14,
'text.usetex': True,
'figure.figsize': (12, 8)})

# Define the Hill activation and repression functions
def hill_activation(x, K, n):
    return x**n / (K**n + x**n)

def hill_repression(x, K, n):
    return K**n / (K**n + x**n)

# Generate x values
x = np.linspace(0, 10, 400)

# Define parameters for different curves
parameters_activation_n = [
    {"K": 2, "n": 1},
    {"K": 2, "n": 2},
    {"K": 2, "n": 4},
    {"K": 2, "n": 8},
]

parameters_activation_K = [
    {"K": 1, "n": 4},
    {"K": 2, "n": 4},
    {"K": 4, "n": 4},
]

parameters_repression_n = [
    {"K": 2, "n": 1},
    {"K": 2, "n": 2},
    {"K": 2, "n": 4},
    {"K": 2, "n": 8},
]

parameters_repression_K = [
    {"K": 1, "n": 4},
    {"K": 2, "n": 4},
    {"K": 4, "n": 4},
]

# Create tight subplot layout
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Plot the effect of changing n for activation
for param in parameters_activation_n:
    y = hill_activation(x, param["K"], param["n"])
    ax[0, 0].plot(x, y, label=f"K={param['K']}, n={param['n']}")
ax[0, 0].set_title(r"Hill Activation: Effect of Changing $n$")
ax[0, 0].set_xlabel(r"$x$")
ax[0, 0].set_ylabel(r"$H_1(x)$")
ax[0, 0].legend()
ax[0, 0].grid(True)

# Plot the effect of changing K for activation
for param in parameters_activation_K:
    y = hill_activation(x, param["K"], param["n"])
    ax[0, 1].plot(x, y, label=f"K={param['K']}, n={param['n']}")
ax[0, 1].set_title(r"Hill Activation: Effect of Changing $K$")
ax[0, 1].set_xlabel(r"$x$")
ax[0, 1].set_ylabel(r"$H_1(x)$")
ax[0, 1].legend()
ax[0, 1].grid(True)

# Plot the effect of changing n for repression
for param in parameters_repression_n:
    y = hill_repression(x, param["K"], param["n"])
    ax[1, 0].plot(x, y, label=f"K={param['K']}, n={param['n']}")
ax[1, 0].set_title(r"Hill Repression: Effect of Changing $n$")
ax[1, 0].set_xlabel(r"$x$")
ax[1, 0].set_ylabel(r"$H_2(x)$")
ax[1, 0].legend()
ax[1, 0].grid(True)

# Plot the effect of changing K for repression
for param in parameters_repression_K:
    y = hill_repression(x, param["K"], param["n"])
    ax[1, 1].plot(x, y, label=f"K={param['K']}, n={param['n']}")
ax[1, 1].set_title(r"Hill Repression: Effect of Changing $K$")
ax[1, 1].set_xlabel(r"$x$")
ax[1, 1].set_ylabel(r"$H_2(x)$")
ax[1, 1].legend()
ax[1, 1].grid(True)

plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/HiiFcuntion_demo.png',dpi=300)
