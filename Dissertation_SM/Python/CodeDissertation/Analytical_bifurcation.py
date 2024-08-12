#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:43:21 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from Functions_HN_model import *
import matplotlib.colors as mcolors
import warnings

# basal degradation rates 
mu_m = np.log(2)/30
mu_p = np.log(2)/90
mu_pn= np.log(2)/40
mu_mn = np.log(2)/46



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
'figure.figsize': (13, 9)})


params = {
    't': np.arange(0, 12000 + 0.1, 0.1),  # Example time array
    'tau1': 50,
    'tau2': 0,
    'p01': 300,
    'p02': 500,
    'SS': 1,
    'activation': False,
    'a_m': 1,
    'a_p': 10,
    'a_mn': 1,
    'a_pn': 4,
    'n01': 5,
    'n02': 5
}
t = params['t']
T = max(t)
params2 = {
    't': np.arange(0, 12000 + 0.1, 0.1),  # Example time array
    'tau1': 200,
    'tau2': 10,
    'p01': 500,
    'p02': 500,
    'SS': 1,
    'activation': False,
    'a_m': 1,
    'a_p': 10,
    'a_mn': 1,
    'a_pn': 4,
    'n01': 2,
    'n02': 5
}


def hill_repression_derivative(x, K, n):
    return - (n * K**n * x**(n-1)) / (K**n + x**n)**2

# equation for the constaint bound
def repression_eq(x):
    return abs(hill_repression_derivative(x,params_tmp['p01'],params_tmp['n01'])) - mu_m*mu_p/params_tmp['a_m']/params_tmp['a_p']


color = ["darkcyan", "darkmagenta", 
    "darkolivegreen", "darkorange", "darkorchid", "darkslateblue", 
    "darkslategray", "darkviolet", "dimgray", "midnightblue", "navy", 
    "saddlebrown"
]
# am = [0.002,0.0043,0.005,0.06,0.1,1]
am = [0.0043,0.06,0.1,1]
am = [0.0043,0.007,0.1,1]
# Create subplots
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()  # Flatten to easily index the subplots
# Labels for subplots
labels = ['(a)', '(b)', '(c)', '(d)']

# Loop through am values and create each subplot
for ii in range(len(am)):
    # if ii == len(am)-1:
    #     params_tmp = params2
    #     x = np.linspace(0, 4000, 1000)
    # else:
    #     params_tmp = params.copy()
    #     x = np.linspace(0, 1200, 1000)
    x = np.linspace(0, 1200, 1000)
    params_tmp = params2.copy()
    params_tmp['a_m'] = am[ii]
    M_hes, P_hes, m_NGN, p_NGN = minimal_model_solver(params_tmp)

    # Analytical
    index = t > T * 0.8
    initial_guess = [np.mean(M_hes[index]), np.mean(P_hes[index]), np.mean(m_NGN[index]), np.mean(p_NGN[index])]
    solutions = fsolve(lambda vars: model(vars,params_tmp), initial_guess)
    axs[ii].plot(solutions[1], G1(solutions[1], params_tmp['n01'], params_tmp['p01']), markersize=20, marker='x', label='The intersect (analytical solution)')

    # Constraint bound
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x_rep1 = abs(fsolve(repression_eq, 0))
            x_rep2 = abs(fsolve(repression_eq, 2 * params_tmp['p01'] - x_rep1))
    except RuntimeWarning:
        x_rep1 = np.nan
        x_rep2 = np.nan

    x_o_bound = np.array([x_rep1, x_rep2])
    axs[ii].plot(x_o_bound, G1(x_o_bound, params_tmp['n01'], params_tmp['p01']), 'o', linewidth=3, color=color[-1], markersize=10, label='The constraint bound')

    # Slope
    slope = mu_m * mu_p / params_tmp['a_m'] / params_tmp['a_p']
    ss = np.linspace(np.min(P_hes[t > 0.8 * max(t)]), np.max(P_hes[t > 0.8 * max(t)]), 60)
    y = G1(x, params_tmp['n01'], params_tmp['p01'])
    axs[ii].plot(x, y, 'black', alpha=0.3, linewidth=4, linestyle='-', label='The Hill function')
    if ss[1] - ss[0] > 1e-2:
        axs[ii].plot(ss, G1(ss, params_tmp['n01'], params_tmp['p01']), linewidth=3, linestyle='--', color=color[ii], alpha=1, label=f'The oscillation range:({np.min(ss):.2f},{np.max(ss):.2f})')
    axs[ii].plot(x, slope * x, color='blue', linewidth=4, alpha=0.4, label=f'y={slope:.2e}x')

    # Formatting
    axs[ii].set_ylim([-0.1, 1.1])
    axs[ii].set_xlabel(r"$x$")
    axs[ii].set_ylabel(r"$H_2(x)$")
    axs[ii].legend()
    axs[ii].grid(True)
    axs[ii].text(0.05, 0.95, labels[ii], transform=axs[ii].transAxes, fontsize=14)
    
plt.tight_layout()
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/HillProbeSubplot2.png',dpi=300)
plt.show()
    
# graphical solution of constraint bound
plt.plot(x,abs(hill_repression_derivative(x,params_tmp['p01'],params_tmp['n01'])))
# y = G1(x, params_tmp['n01'],params_tmp['p01'])
# plt.plot(x,y,'black',alpha=0.1,linewidth=3,linestyle='--')
plt.axhline(mu_m*mu_p/params_tmp['a_m']/params_tmp['a_p'])

    
    