#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:02:33 2024

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
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

# ste up 
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
    
    
x1 = np.linspace(2,7,120)
x2 = np.linspace(30,140,120)

# Create a meshgrid for alpha and beta
x1_grid, x2_grid = np.meshgrid(x1, x2)

amp_grid = np.zeros_like(x1_grid)
period_grid = np.zeros_like(x1_grid)
corr_grid = np.zeros_like(x1_grid)
corr2_grid = np.zeros_like(x1_grid)
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        params_tmp = params.copy()
        params_tmp['n01']=x1_grid[i,j]
        params_tmp['tau1']=x2_grid[i,j]
        period_grid[i,j],amp_grid[i,j],corr_grid[i,j], corr2_grid[i,j]= period_fold(params_tmp)
        


# Normalize the period and fold
v1_grid,v2_grid = x1_grid,x2_grid
v1,v2 = x1,x2
norm_period = Normalize(vmin=np.nanmin(period_grid.flatten()), vmax=np.nanmax(period_grid.flatten()))
norm_amp = Normalize(vmin=np.nanmin(amp_grid.flatten()), vmax=np.nanmax(amp_grid.flatten()))
norm_corr = Normalize(vmin=np.nanmin(corr_grid.flatten()), vmax=np.nanmax(corr_grid.flatten()))


# plot corr
colors_corr = plt.cm.plasma(norm_corr(corr_grid))
# Modify the alpha channel of colors based on foldvalues
scaling_p = 1
alpha_values = norm_period(period_grid)
alpha_values = np.power(alpha_values, scaling_p) 

# Apply alpha values to the colors array
# colors_corr[..., -1] = np.ones_like(alpha_values) # Set the alpha channel

# plots
fig, ax = plt.subplots()
ax.contour(v1_grid,v2_grid, amp_grid,levels=[1.47], colors='red',linewidths=1.2,alpha=1,linestyles='--')
contour = ax.contour(v1_grid,v2_grid, period_grid,levels=[np.nanmin(period_grid.flatten())+0.1,4.6], colors='blue',linewidths=1.2,alpha=1,linestyles='--')
ax.contour(v1_grid,v2_grid, corr_grid,levels=[0], colors='black',linewidths=1.5,alpha=0.8,linestyles='-')

# Plot using the modified color array with transparency
img = ax.imshow(colors_corr, extent=(v1.min(), v1.max(), v2.min(), v2.max()), origin='lower', aspect='auto',alpha=0.9)

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm_corr)
sm.set_array([])  # Set an empty array to link the colorbar with the colormap

# Directly pass the ScalarMappable object to plt.colorbar and link to the axis
cbar = plt.colorbar(sm, ax=ax, label='Correlation')
cbar.ax.tick_params(labelsize=10)
# Aesthetic improvements
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$n_{01}$')
plt.ylabel(r'$\tau_1$')
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/CorrParamterImshow.png',dpi=300)


# Define more transparent layers for the legend
fold_legend =  (period_grid.max()-period_grid.min())*np.linspace(0,1,3)**scaling_p + period_grid.min()
patches = [mpatches.Patch(color='black', alpha=np.power(norm_period(val), 1), label=f'{val:.2f}') for val in fold_legend]

# Add the legend to the plot
plt.legend(handles=patches, loc='lower left', title="Period", fontsize=10)

# Aesthetic improvements
ax.grid(True, linestyle='--', linewidth=0.5)
ax.xlabel(r'$n_{01}$')
ax.ylabel(r'$\tau_1$')
ax.tight_layout()
# ax.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/CorrParamter.png',dpi=300)

# corr2
# Normalize the period and fold
v1_grid,v2_grid = x1_grid,x2_grid
v1,v2 = x1,x2
norm_period = Normalize(vmin=np.nanmin(period_grid.flatten()), vmax=np.nanmax(period_grid.flatten()))
norm_amp = Normalize(vmin=np.nanmin(amp_grid.flatten()), vmax=np.nanmax(amp_grid.flatten()))
norm_corr2 = Normalize(vmin=np.nanmin(corr2_grid.flatten()), vmax=np.nanmax(corr2_grid.flatten()))


# plot corr
colors_corr2 = plt.cm.plasma(norm_corr2(corr2_grid))
# Modify the alpha channel of colors based on foldvalues
scaling_p = 1
alpha_values = norm_period(period_grid)
alpha_values = np.power(alpha_values, scaling_p) 

# Apply alpha values to the colors array
# colors_corr[..., -1] = np.ones_like(alpha_values) # Set the alpha channel

# plots
fig, ax = plt.subplots()
ax.contour(v1_grid,v2_grid, amp_grid,levels=[1.47], colors='red',linewidths=1.2,alpha=1,linestyles='--')
contour = ax.contour(v1_grid,v2_grid, period_grid,levels=[np.nanmin(period_grid.flatten())+0.1,4.6], colors='blue',linewidths=1.2,alpha=1,linestyles='--')
ax.contour(v1_grid,v2_grid, corr2_grid,levels=[0], colors='black',linewidths=1.5,alpha=0.8,linestyles='-')

# Plot using the modified color array with transparency
img = ax.imshow(colors_corr2, extent=(v1.min(), v1.max(), v2.min(), v2.max()), origin='lower', aspect='auto',alpha=0.9)

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm_corr2)
sm.set_array([])  # Set an empty array to link the colorbar with the colormap

# Directly pass the ScalarMappable object to plt.colorbar and link to the axis
cbar = plt.colorbar(sm, ax=ax, label='Correlation')
cbar.ax.tick_params(labelsize=10)
# Aesthetic improvements
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$n_{01}$')
plt.ylabel(r'$\tau_1$')
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/CorrParamterImshowCorr2.png',dpi=300)




# phase according to experimental data
np.arctan(-np.pi*2/2/60/mu_mn)+np.arctan(-np.pi*2/2/60/mu_pn)
np.arctan(-np.pi*2/4.6/60/mu_mn)+np.arctan(-np.pi*2/4.6/60/mu_pn)

# Plot the data separately
plt.figure(2)
plt.contourf(v1_grid,v2_grid, corr_grid, levels=20, cmap='viridis', norm=norm_corr,alpha=0.7,linewidths=0.1)
# contour = plt.contour(v1_grid,v2_grid, amp_grid,levels=[0.01], colors='black',linewidths=1.2,alpha=0.1,linestyles='-')
plt.colorbar(label='Correlation')
plt.contour(v1_grid,v2_grid, period_grid,levels=[np.nanmin(period_grid.flatten())+0.1,4.6], colors='Blue',linewidths=1.2,alpha=0.8,linestyles='--')
plt.contour(v1_grid,v2_grid, amp_grid,levels=[1.47], colors='red',linewidths=1.2,alpha=0.8,linestyles='--')
plt.contour(v1_grid,v2_grid, corr_grid,levels=[0], colors='Black',linewidths=1.5,alpha=0.8,linestyles='-')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$n_{01}$')
plt.ylabel(r'$\tau_1$')
plt.tight_layout()
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/CorrParamter.png',dpi=300)
plt.show()

   
plt.figure(3)
# Overlay period as a contour with darkness representing the period
plt.contourf(v1_grid,v2_grid,period_grid,levels=20,alpha=0.8,linewidth=0,norm=norm_period)
plt.colorbar(label='Period')
plt.contour(v1_grid,v2_grid, period_grid,levels=[np.nanmin(period_grid.flatten())+0.1,4.6], colors='Blue',linewidths=1.2,alpha=1,linestyles='--')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$n_{01}$')
plt.ylabel(r'$\tau_1$')
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/PeriodParamter1.png',dpi=300)


plt.figure(4)
plt.contourf(v1_grid,v2_grid,amp_grid,levels=20,alpha=0.8,norm = norm_amp)
plt.colorbar(label='Fold change')
plt.contour(v1_grid,v2_grid, amp_grid,levels=[1.47], colors='Blue',linewidths=1.2,alpha=1,linestyles='--')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel(r'$n_{01}$')
plt.ylabel(r'$\tau_1$')
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/FoldParamter1.png',dpi=300)
























plt.contour(v1_grid,v2_grid, period_grid,levels=[2,4.6], colors='Black',linewidths=1.2,alpha=0.8,linestyles='--')
plt.contour(v1_grid,v2_grid, amp_grid,levels=[1.47], colors='red',linewidths=1.2,alpha=0.8,linestyles='--')



plt.contourf(v1_grid,v2_grid, np.arctan(2*np.pi/period_grid*60/(np.log(2)/mu_mn)) + np.arctan(2*np.pi/period_grid*60/(np.log(2)/mu_pn)),levels=20,linewidth=1,alpha=0.6)
plt.contour(v1_grid,v2_grid, np.arctan(2*np.pi/period_grid*60/(np.log(2)/mu_mn)) + np.arctan(2*np.pi/period_grid*60/(np.log(2)/mu_pn)),levels=[1.7],linewidth=1,alpha=0.6)
plt.colorbar(label='period')

plt.xlabel('Bifurcation Parameter')
plt.ylabel('Bifurcation Parameter')
plt.title('System Solution Properties Against Bifurcation Parameters')
plt.show()
    

np.corrcoef(np.sin(np.linspace(0,3.14,600)),1 + np.sin(np.linspace(0,3.14,600)+1.1))
np.corrcoef(np.sin(np.linspace(0,3.14,600)),4 + 2*np.sin(np.linspace(0,3.14,600)+1.7))

solution_plot(params_tmp,P_hes,p_NGN,M_hes,m_NGN,y_range=[0,np.max(np.vstack((p_NGN[t>T*0.8],P_hes[t>T*0.8])))+600],peakplot=True)
plt.xlim([100,140])
np.corrcoef(P_hes,p_NGN)
