#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:20:25 2024

@author: del
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from Functions_HN_model import *

params = {
    't': np.arange(0, 4000 + 0.01, 0.01),  # Example time array
    'tau1': 10,
    'tau2': 10,
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

plt.rcParams.update({
'font.family': 'serif',
'font.serif': ['Times New Roman'],
'axes.titleweight': 'bold',  # Bold title
'axes.labelweight': 'bold',  # Bold labels
'font.size': 14,
'axes.titlesize': 16,
'axes.labelsize': 16,
'xtick.labelsize': 14,
'ytick.labelsize': 14,
'legend.fontsize': 8,
'text.usetex': True,
'figure.figsize': (6,4)})

# Reference solution with very small dt
dt_ref = 1e-3
params_tmp = params.copy()
params_tmp['t'] = np.arange(0, 4000 + dt_ref, dt_ref)
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
y_ref = M_hes[-1]
# solution_plot(params_tmp,P_hes,p_NGN,M_hes,m_NGN)

# Parameters for testing different dt values
dt_values = np.linspace(dt_ref,10,300)
errors = []

# Calculate errors for different dt
for dt in dt_values:
    params_tmp = params.copy()
    params_tmp['t'] = np.arange(0, 4000 + dt, dt)
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    error = np.abs(M_hes[-1] - y_ref).max()
    errors.append(error)

# Log-log plot
p,_ = find_peaks(-np.array(errors))
plt.loglog(dt_values, np.array(errors), '-o',markersize=3)
plt.axvline(dt_values[p[0]],color='red',linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Errors')
plt.grid(True)
plt.tight_layout() 
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/dterror.png',dpi=300)
