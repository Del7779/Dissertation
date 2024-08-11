#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:14 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve, minimize
from Functions_HN_model import *
import random
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

params = {
    't': np.arange(0, 14000 + 0.09, 0.09),  # Example time array
    'tau1': 45,
    'tau2': 0,
    'p01': 200,
    'p02': 500,
    'SS': 1,
    'activation': False,
    'a_m': 1, # 1 - 10
    'a_p': 10, # 5 - 20
    'a_mn': 1,
    'a_pn': 4,
    'n01': 5,
    'n02': 5
}
t = params['t']
T = max(t)
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

def period_fold(param):
    t = param['t']
    T = max(t)
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(param)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    fold_hes = steady_hes[-1]/steady_hes[-2]
    fold_ngn = steady_ngn[-1]/steady_ngn[-2]
    if any(np.array([fold_hes,fold_ngn])-1 < 1e-3):
        return np.nan,np.nan,np.nan,np.nan
    else:
        corr1 = np.corrcoef(P_hes[t>0.8*T],p_NGN[t>0.8*T])[0,1]
        corr2 = np.corrcoef(p_NGN[t>0.8*T],G1(P_hes[t>0.8*T],n=param['n02'],p0=param['p02']))[0,1]
        pks_hes,_ = find_peaks(P_hes[t>T*0.8])
        pks_ngn,_ = find_peaks(p_NGN[t>T*0.8])
        period_hes = np.mean(np.diff(t[t>T*0.8][pks_hes]))/60
        period_ngn = np.mean(np.diff(t[t>T*0.8][pks_ngn]))/60
        
        return period_hes,fold_hes,corr1,corr2
    
    
x1 = np.linspace(0.0001,10,120)
x2 = np.linspace(30,120,120)

# Create a meshgrid for alpha and beta
x1_grid, x2_grid = np.meshgrid(x1, x2)

amp_grid = np.zeros_like(x1_grid)
period_grid = np.zeros_like(x1_grid)
corr_grid = np.zeros_like(x1_grid)
corr2_grid = np.zeros_like(x1_grid)
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        params_tmp = params.copy()
        params_tmp['a_m']=x1_grid[i,j]
        params_tmp['tau1']=x2_grid[i,j]
        period_grid[i,j],amp_grid[i,j],corr_grid[i,j], corr2_grid[i,j]= period_fold(params_tmp)
        
# Normalize the period and fold
v1_grid = mu_m*mu_p/params['a_p']/x1_grid
v1 = mu_m*mu_p/params['a_p']/x1
v2_grid = x2_grid
v2 = x2
norm_period = Normalize(vmin=np.nanmin(period_grid1.flatten()), vmax=np.nanmax(period_grid1.flatten()))
norm_amp = Normalize(vmin=np.nanmin(amp_grid1.flatten()), vmax=np.nanmax(amp_grid1.flatten()))
norm_corr = Normalize(vmin=np.nanmin(corr_grid.flatten()), vmax=np.nanmax(corr_grid.flatten()))



# Create the color array based on the period values using the 'plasma' colormap
colors = plt.cm.viridis(norm_period(period_grid1))

# Create a figure and axis
fig, ax = plt.subplots()

ax.contour(v1_grid,v2_grid, period_grid,levels=[2,4.6], colors='Blue',linewidths=2,alpha=0.9,linestyles='--')
# Plot using the modified color array with transparency
img = ax.imshow(colors, extent=(v1.min(), v1.max(), v2.min(), v2.max()), origin='lower', aspect='auto',alpha=0.8)

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_period)
sm.set_array([])  # Set an empty array to link the colorbar with the colormap

# Directly pass the ScalarMappable object to plt.colorbar and link to the axis
cbar = plt.colorbar(sm, ax=ax, label='Period')
cbar.ax.tick_params(labelsize=10)

# Aesthetic improvements
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$\frac{\mu_{m_1} \mu_{p_1}}{\alpha_{m_1} \alpha_{p_1}}$')
plt.ylabel(r'$\tau_1$')
plt.show()
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/PeriodParamterslope.png',dpi=300)







    
    
# Create the color array based on the period values using the 'plasma' colormap
colors = plt.cm.viridis(norm_amp(amp_grid1))

# Create a figure and axis
fig, ax = plt.subplots()

ax.contour(v1_grid,v2_grid, amp_grid1,levels=[20], colors='Blue',linewidths=2,alpha=0.9,linestyles='--')
# Plot using the modified color array with transparency
img = ax.imshow(colors, extent=(v1.min(), v1.max(), v2.min(), v2.max()), origin='lower', aspect='auto',alpha=0.8)

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_amp)
sm.set_array([])  # Set an empty array to link the colorbar with the colormap

# Directly pass the ScalarMappable object to plt.colorbar and link to the axis
cbar = plt.colorbar(sm, ax=ax, label='Fold change')
cbar.ax.tick_params(labelsize=10)

# Aesthetic improvements
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$\frac{\mu_{m_1} \mu_{p_1}}{\alpha_{m_1} \alpha_{p_1}}$')
plt.ylabel(r'$\tau_1$')
plt.show()
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/FoldParamterslope.png',dpi=300)

    