#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from Functions_HN_model import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



params = {
    't': np.arange(0, 14000 + 0.09, 0.09),  # Example time array
    'tau1': 42,
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
dt = t[2] - t[1]
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
'legend.fontsize':9,
'text.usetex': True,
'figure.figsize': (6,4)})

M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params)
solution_plot(params,P_hes,p_NGN,M_hes,m_NGN,aplot=True)

# fold change : NGN2 2.45 HES5 1.47

# P02
P02_loop = np.linspace(1,2500,70)
P01_loop = np.linspace(1,2500,70)
steady_hes_loop = np.zeros((2,len(P01_loop)))
steady_ngn_loop = np.zeros((2,len(P01_loop)))
period_loop = np.zeros((2,len(P01_loop)))
fold_loop = np.zeros_like(steady_hes_loop)
for ii in range(len(P02_loop)):
    params_tmp = params.copy()
    params_tmp['p02'] = P02_loop[ii]
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    ##
    peaks, _ = find_peaks(P_hes)
    period1 = np.mean(np.diff(t[peaks[-3:-1]]))
    period_loop[0,ii] = period1/60
    peaks, _ = find_peaks(p_NGN)
    period2 = np.mean(np.diff(t[peaks[-3:-1]]))   
    period_loop[1,ii] = period2/60  
    ##
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    steady_hes_loop[:,ii] = steady_hes.T
    steady_ngn_loop[:,ii] = steady_ngn.T
    fold_loop[0,ii]= steady_hes[-1]/steady_hes[0]
    fold_loop[1,ii]= steady_ngn[-1]/steady_ngn[0]
    

plt.figure(1)
plt.plot(P02_loop,steady_ngn_loop[0,:],'r.',label='Ngn2')
plt.xlabel(r'Repression Threshold $p_{02}$')
plt.ylabel('NGN2 Steady State')
plt.plot(P02_loop,steady_ngn_loop[1,:],'r.',label='Hes5')
plt.xlabel(r'Repression Threshold $p_{02}$')
plt.ylabel('Ngn2 Steady State')
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/p02bifurcation.png',dpi=300)

plt.figure(2)
plt.plot(P02_loop,fold_loop[0,:],'r.',label='Hes5',)
plt.plot(P02_loop,fold_loop[1,:],'b.',label='Nng2')
plt.axhline(1.47,color='red',linestyle='--')
plt.axhline(2.45,color='blue',linestyle='--')
plt.ylabel('Expression fold change')
plt.xlabel(r'Repression Threshold $p_{02}$')
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/p02foldchange.png',dpi=300)

 

# P01
P01_loop = np.linspace(1,2500,70)
steady_hes_loop = np.zeros((2,len(P01_loop)))
steady_ngn_loop = np.zeros((2,len(P01_loop)))
fold_loop = np.zeros_like(steady_hes_loop)
period_loop = np.zeros((2,len(P01_loop)))
tau_tmp = [45,50]
for jj in range(len(tau_tmp)):
    params_tmp = params.copy()
    params_tmp['tau1'] = tau_tmp[jj]
    for ii in range(len(P01_loop)): 
        params_tmp['p01'] = P01_loop[ii]
        M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
        peaks, _ = find_peaks(P_hes)
        period1 = np.mean(np.diff(t[peaks[-3:-1]]))
        period_loop[0,ii] = period1/60
        peaks, _ = find_peaks(p_NGN)
        period2 = np.mean(np.diff(t[peaks[-3:-1]]))   
        period_loop[1,ii] = period2/60 
        steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
        steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
        steady_hes_loop[:,ii] = steady_hes.T
        steady_ngn_loop[:,ii] = steady_ngn.T
        fold_loop[0,ii]= steady_hes[-1]/steady_hes[0]
        fold_loop[1,ii]= steady_ngn[-1]/steady_ngn[0]
        
    
    plt.figure(1,figsize=(6,4))
    plt.plot(P01_loop,steady_ngn_loop[0,:],'r.',label=f'NGN2 delay:{tau_tmp[jj]} minutes)',color=[jj/len(tau_tmp),0.3,0])
    plt.xlabel(r'Repression threshold $p_{01}$')
    plt.ylabel('Ngn2 steady oscillation')
    plt.plot(P01_loop,steady_ngn_loop[1,:],'r.',color=[jj/len(tau_tmp),0.3,0])
    plt.xlabel(r'Repression threshold $p_{01}$')
    plt.ylabel('Ngn steady oscillation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/p01bifurcation.png',dpi=300)
    
    plt.figure(2,figsize=(6,4))
    plt.plot(P01_loop,fold_loop[0,:],'r.',label=f'HES5:(delay:{tau_tmp[jj]} minutes)',color=[jj/len(tau_tmp),0.5,0])
    plt.plot(P01_loop,fold_loop[1,:],'b.',label=f'NGN2:(delay:{tau_tmp[jj]} minutes)',color=[0,0.3,jj/len(tau_tmp)])


plt.figure(2)
plt.axhline(1.47,color='red',linestyle='--',label='Hes5(experiment)')
plt.axhline(2.45,color='blue',linestyle='--',label='Ngn2(experiment)')
plt.ylabel('Expression fold change')
plt.xlabel(r'Repression threshold $p_{01}$')
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/p01foldchange.png',dpi=300)

## P01 should not exceed 500

# tau1
tau1_loop = np.linspace(10,100,80)
tau2_loop = np.linspace(10,100,80)
steady_hes_loop = np.zeros((2,len(tau1_loop)))
steady_ngn_loop = np.zeros((2,len(tau1_loop)))
fold_loop = np.zeros_like(steady_hes_loop)
for ii in range(len(tau1_loop)):
    params_tmp = params
    params_tmp['p01'],params_tmp['tau1'] = 300, tau1_loop[ii]
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    steady_hes_loop[:,ii] = steady_hes.T
    steady_ngn_loop[:,ii] = steady_ngn.T
    fold_loop[0,ii]= steady_hes[-1]/steady_hes[0]
    fold_loop[1,ii]= steady_ngn[-1]/steady_ngn[0]
    
plt.figure(1,figsize=(6,4))
plt.plot(tau1_loop,steady_ngn_loop[0,:],'b.',label='Ngn2')
plt.xlabel(r'$\tau_1$')
plt.ylabel('Ngn2 steady oscillation')
plt.plot(tau1_loop,steady_ngn_loop[1,:],'b.',label='Ngn2')
plt.xlabel(r'$\tau_1$')
plt.ylabel('Ngn2 steady oscillation')
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau1osc.png',dpi=300)

plt.figure(2,figsize=(6,4))
plt.plot(tau1_loop,fold_loop[0,:],'r.',label='Hes5',)
plt.plot(tau1_loop,fold_loop[1,:],'b.',label='Ngn2')
plt.axhline(1.47,color='red',linestyle='--')
plt.axhline(2.45,color='blue',linestyle='--')
plt.ylabel('Expression fold change')
plt.xlabel(r'$\tau_1$')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau1foldchange2.png',dpi=300)


# Create the main figure and axis
fig, ax_main = plt.subplots(figsize=(10, 6))

# Plot the main figure (equivalent to figure 2)
ax_main.plot(tau1_loop, fold_loop[0, :], 'r.', label='HES5')
ax_main.plot(tau1_loop, fold_loop[1, :], 'b.', label='NGN2')
ax_main.axhline(1.47, color='red', linestyle='--')
ax_main.axhline(2.45, color='blue', linestyle='--')
ax_main.set_ylabel('Expression fold change')
ax_main.set_xlabel(r'$\tau_1$')
ax_main.legend()

# Create the inset axis in the left top corner
ax_inset = inset_axes(ax_main, width="100%", height="80%", loc="upper left",bbox_to_anchor=(0.1, 0.2, 0.4, 0.6),bbox_transform=ax_main.transAxes)

# Plot the inset figure (equivalent to figure 1)
ax_inset.plot(tau1_loop, steady_ngn_loop[0, :], 'r.', label='NGN2')
ax_inset.plot(tau1_loop, steady_ngn_loop[1, :], 'r.')
ax_inset.set_xlabel(r'$\tau_1$')
ax_inset.set_ylabel('Ngn2 steady oscillation')
ax_inset.grid(True)

# Adjust layout to fit everything
plt.tight_layout()

# Save the figure with bounding box tight
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau1foldchange.png',dpi=300)

# Display the figure
plt.show()

# conclusion 

# tau2
tau2_loop = np.linspace(10,100,80)
steady_hes_loop = np.zeros((2,len(tau2_loop)))
steady_ngn_loop = np.zeros((2,len(tau2_loop)))
fold_loop = np.zeros_like(steady_hes_loop)
for ii in range(len(tau2_loop)):
    params_tmp = params
    params_tmp['tau1'] = 45

    params_tmp['tau2'] = tau2_loop[ii]
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    steady_hes_loop[:,ii] = steady_hes.T
    steady_ngn_loop[:,ii] = steady_ngn.T
    fold_loop[0,ii]= steady_hes[-1]/steady_hes[0]
    fold_loop[1,ii]= steady_ngn[-1]/steady_ngn[0]
    
plt.figure(1,figsize=(6,4))
plt.plot(tau2_loop,steady_ngn_loop[0,:],'b.',label='Ngn2')
plt.xlabel(r'$\tau_2$')
plt.ylabel('Ngn2 steady oscillation')
plt.plot(tau2_loop,steady_ngn_loop[1,:],'b.',label='Ngn2')
plt.xlabel(r'$\tau_2$')
plt.ylabel('Ngn2 steady oscillation')
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau2osc.png',dpi=300)

plt.figure(2,figsize=(6,4))
plt.plot(tau2_loop,fold_loop[0,:],'r.',label='Hes5',)
plt.plot(tau2_loop,fold_loop[1,:],'b.',label='Ngn2')
plt.axhline(1.47,color='red',linestyle='--')
plt.axhline(2.45,color='blue',linestyle='--')
plt.ylabel('Expression fold change')
plt.xlabel(r'$\tau_2$')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau2foldchange2.png',dpi=300)

# Hill coefficient
n_loop = np.linspace(1,7,100)


steady_hes_loop = np.zeros((2,len(n_loop)))
steady_ngn_loop = np.zeros((2,len(n_loop)))
fold_loop = np.zeros_like(steady_hes_loop)
for ii in range(len(n_loop)):
    params_tmp = params
    params_tmp['n01'] = n_loop[ii]
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    steady_hes_loop[:,ii] = steady_hes.T
    steady_ngn_loop[:,ii] = steady_ngn.T
    fold_loop[0,ii]= steady_hes[-1]/steady_hes[0]
    fold_loop[1,ii]= steady_ngn[-1]/steady_ngn[0]
    

    
plt.figure(1,figsize=(6,4))
plt.plot(n_loop,steady_ngn_loop[0,:],'b.',label='Ngn2')
plt.xlabel(r'$n_2$')
plt.ylabel('Ngn2 steady oscillation')
plt.plot(n_loop,steady_ngn_loop[1,:],'b.',label='Ngn2')
plt.xlabel(r'$n_0$')
plt.ylabel('Ngn2 steady oscillation')
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau2osc.png',dpi=300)

plt.figure(2,figsize=(6,4))
plt.plot(n_loop,fold_loop[0,:],'r.',label='Hes5',)
plt.plot(n_loop,fold_loop[1,:],'b.',label='Ngn2')
plt.axhline(1.47,color='red',linestyle='--')
plt.axhline(2.45,color='blue',linestyle='--')
plt.ylabel('Expression fold change')
plt.xlabel(r'$n_0$')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure with bounding box tight
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tau2foldchange2.png',dpi=300)








params = {
    't': np.arange(0, 20000 + 0.09, 0.09),  # Example time array
    'tau1': 42,
    'tau2': 0,
    'p01': 200,
    'p02': 500,
    'SS': 1,
    'activation': False,
    'a_m': 1, # 1 - 10
    'a_p': 10, # 5 - 20
    'a_mn': 10,
    'a_pn': 10,
    'n01': 5,
    'n02': 5
}

t = params['t']
T = max(t)
dt = t[2] - t[1]
params_tmp = params.copy()
solution = result1[37].x
params_tmp['a_m'], params_tmp['a_p'], params_tmp['p01'],params_tmp['p02'],params_tmp['tau1'],params_tmp['n01'],params_tmp['n02'] = tuple(solution)
params = params_tmp.copy()
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params)
solution_plot(params,P_hes,p_NGN,M_hes,m_NGN,y_range=[0,10000],figname='OptimisedSolution')


# params = {
#     't': np.arange(0, 14000 + 0.09, 0.09),  # Example time array
#     'tau1': 40,
#     'tau2': 0,
#     'p01': 533,
#     'p02': 747,
#     'SS': 1,
#     'activation': False,
#     'a_m': 2, # 1 - 10
#     'a_p': 10, # 5 - 20
#     'a_mn': 4,
#     'a_pn': 4,
#     'n01': 4,
#     'n02': 6.8
# }
# t = params['t']
# T = max(t)
# dt = t[2] - t[1]







# conclusion 
# tau2 can't alter the fold of both gene expressions


# correlation analysis tau2 
tau2_loop = np.linspace(0,200,80)
period_loop = np.zeros((2,len(tau2_loop)))
lagpeak_loop = np.zeros_like(tau2_loop)
corm_loop2 = np.zeros_like(tau2_loop)
cor0_loop2 = np.zeros_like(tau2_loop)
for jj in range(len(tau2_loop)):
    params_tmp = params.copy()
    params_tmp['tau2'] = tau2_loop[jj]
    
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    ##
    peaks, _ = find_peaks(P_hes)
    period1 = np.mean(np.diff(t[peaks[-5:-2]]))
    period_loop[0,jj] = period1/60
    peaks, _ = find_peaks(p_NGN)
    period2 = np.mean(np.diff(t[peaks[-5:-2]]))   
    period_loop[1,jj] = period2/60 
    period = (period1+period2)/2
    
    # index = (t>period*15) & (t<period*17)
    # lags, cc = cross_correlation(p_NGN[index],P_hes[index],int(sum(index)/2))
    lags, cc, lagshift = cross_correlation(p_NGN[-int(2*period/dt):-1],P_hes[-int(2*period/dt):-1],int(2*period/dt/2))
    lags = lags*dt*lagshift
    peaks,_ = find_peaks(cc)
    valleys,_ = find_peaks(-cc)
    near0peak_ind = peaks[np.argmin(abs(lags[peaks]))]
    near0valley_ind = valleys[np.argmin(abs(lags[valleys]))]
    if abs(lags[near0peak_ind]/60) > abs(lags[near0valley_ind]/60):
        lagpeak_loop[jj] = abs(lags[near0valley_ind]/60)
        corm_loop2[jj] = cc[near0valley_ind]
    else:
        lagpeak_loop[jj] = abs(lags[near0peak_ind]/60)
        corm_loop2[jj] = cc[near0peak_ind]
        
    cor0_loop2[jj] = cc[0]

plt.figure(3)
plt.plot(lags/60,cc)
plt.xlabel('Lag')
plt.ylabel('The pearson correlation')
plt.grid(True)
plt.tight_layout()
# plt.title(f'lag period {p}')
plt.legend()
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/crosscorlag.png',dpi=300)


plt.figure(4)
peaks,_ = find_peaks(-lagpeak_loop)
p = np.mean(np.diff(tau2_loop[peaks]))/60
p_lag_ind = (np.sign(corm_loop2)>0)
n_lag_ind = (np.sign(corm_loop2)<0)
plt.plot(tau2_loop[p_lag_ind]/60,lagpeak_loop[p_lag_ind],'r.',label='Positive Correlation')
plt.plot(tau2_loop[n_lag_ind]/60,lagpeak_loop[n_lag_ind],'b.',label='Negative Correlation')
plt.ylabel('Lag/hrs')
plt.xlabel(r'$\tau_2$/hrs')
plt.tight_layout()
# plt.title(f'lag period {p}')
plt.legend()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/crosscortau2.png',dpi=300)

plt.figure(2)
plt.plot(tau2_loop/60,cor0_loop2)
plt.xlabel(r'$\tau_2$/hrs')
plt.ylabel('The pearson correlation')
plt.grid(True)
plt.tight_layout()
# plt.title(f'lag period {p}')
plt.legend()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/crosscor0tau2.png',dpi=300)

 

# conclusion: Period has a sharp but sutle transition 


# correlation analysis tau1 
tau1_loop = np.linspace(70,120,50)
period_loop = np.zeros((2,len(tau1_loop)))
lagpeak_loop = np.zeros_like(tau1_loop)
corm_loop = np.zeros_like(tau1_loop)
cor0_loop = np.zeros_like(tau1_loop)

for jj in range(len(tau1_loop)):
    params_tmp= params.copy()
    params_tmp['tau1'] = tau1_loop[jj]
    
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    
    ## period
    peaks, _ = find_peaks(P_hes)
    period1 = np.mean(np.diff(t[peaks[-5:-2]]))
    period_loop[0,jj] = period1/60
    peaks, _ = find_peaks(p_NGN)
    period2 = np.mean(np.diff(t[peaks[-5:-2]]))   
    period_loop[1,jj] = period2/60 
    period = (period1+period2)/2
    
    # index = (t>period*15) & (t<period*17)
    # lags, cc = cross_correlation(p_NGN[index],P_hes[index],int(sum(index)/2))
    lags, cc, lagshift = cross_correlation(p_NGN[-int(2*period/dt):-1],P_hes[-int(2*period/dt):-1],int(2*period/dt/2))
    lags = lags*dt*lagshift
    peaks,_ = find_peaks(cc)
    valleys,_ = find_peaks(-cc)
    near0peak_ind = peaks[np.argmin(abs(lags[peaks]))]
    near0valley_ind = valleys[np.argmin(abs(lags[valleys]))]
    if abs(lags[near0peak_ind]/60) > abs(lags[near0valley_ind]/60):
        lagpeak_loop[jj] = abs(lags[near0valley_ind]/60)
        corm_loop[jj] = cc[near0valley_ind]
    else:
        lagpeak_loop[jj] = abs(lags[near0peak_ind]/60)
        corm_loop[jj] = cc[near0peak_ind]
    cor0_loop[jj] = cc[0]

plt.figure(3)
plt.plot(lags/60,cc)
plt.xlabel('lag')
plt.ylabel('Pearson Correlation')
plt.grid(True)

plt.figure(4)
peaks,_ = find_peaks(-lagpeak_loop)
p = np.mean(np.diff(tau1_loop[peaks]))/60
p_lag_ind = (np.sign(corm_loop)>0)
n_lag_ind = (np.sign(corm_loop)<0)
plt.plot(tau1_loop[p_lag_ind]/60,lagpeak_loop[p_lag_ind],'r.',label='Positive Correlation')
plt.plot(tau1_loop[n_lag_ind]/60,lagpeak_loop[n_lag_ind],'b.',label='Negative Correlation')
plt.ylabel('Lag/hrs')
plt.xlabel(r'$\tau_1$/hrs')
plt.tight_layout()
# plt.title(f'lag period {p}')
plt.legend()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/crosscortau1.png',dpi=300)


plt.figure(5)
plt.plot(tau1_loop/60,cor0_loop)
plt.xlabel(r'$\tau_1$/hrs')
plt.ylabel('The pearson correlation')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/crosscor0tau1.png',dpi=300)

solution_plot(params,P_hes,p_NGN,M_hes,m_NGN,aplot=True,peakplot=False)

