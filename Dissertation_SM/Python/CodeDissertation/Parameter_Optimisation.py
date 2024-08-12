#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:32:29 2024

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

params = {
    't': np.arange(0, 20000 + 0.09, 0.09),  # Example time array
    'tau1': 45,
    'tau2': 0,
    'p01': 200,
    'p02': 500,
    'SS': 1,
    'activation': False,
    'a_m': 1, # 1 - 10
    'a_p': 10, # 5 - 20
    'a_mn': 3,
    'a_pn': 10,
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

# The fold change of NGN2 is independet of a_mn and a_np
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params)
solution_plot(params,P_hes,p_NGN,M_hes,m_NGN,aplot=True,peakplot=True)


# Define the objective function with multiple variables
def objective_function(vars):
    
    # p01,p02,tau1,n01,n02 = vars
    params_tmp = params.copy()
    params_tmp['a_m'], params_tmp['a_p'], params_tmp['p01'],params_tmp['p02'],params_tmp['tau1'],params_tmp['n01'],params_tmp['n02'] = vars
    M_hes,P_hes,m_NGN,p_NGN = M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    fold_hes = steady_hes[-1]/steady_hes[-2]
    fold_ngn = steady_ngn[-1]/steady_ngn[-2]
    if any(np.array([fold_hes,fold_ngn])-1 < 1e-3):
        return 1e15
    # period
    pks_hes,_ = find_peaks(P_hes[t>T*0.8])
    pks_ngn,_ = find_peaks(p_NGN[t>T*0.8])
    period_hes = np.mean(np.diff(t[t>T*0.8][pks_hes]))
    period_ngn = np.mean(np.diff(t[t>T*0.8][pks_ngn]))
    # fold change : NGN2 2.45 HES5 1.47
    eq1 = 1.47 - fold_hes
    eq2 = 2.45 - fold_ngn

    # if abs(period_hes/60-(3.3+4.93)/2)>(1.3+1.08)/2:
    #     return 1e15
    # if abs(period_hes/60-3.3)>1.3:
    #     return 1e15
    # return np.sum(eq1**2+eq2**2) + params_tmp['n02']**2
    # else:   
    return np.sum(eq1**2+eq2**2)

# Initial guess for the variables
# initial_guess = [1,10,300,500,45,5,5]
# s1

# solution = np.array([1.00000000e+00, 5.00000000e+00, 
#                       1.50000000e+03, 6.42930426e+02,9.83248605e+01, 3.00000000e+00, 3.16456936e+00])

# solution = np.array([1.00000000e+00, 1.46555715e+01, 1.17598179e+03, 1.12482800e+03,8.43776275e+01, 3.00000000e+00, 5.01528629e+00])


# Set up different initial guesses
bounds = [(1,10), (5,20), (500, 1500), (500,1500),(40,200),(3,7),(3,7)]
num_guesses = 50
result1 = []
ing_loop = []
np.random.seed(500)
for ii in range(num_guesses):
    initial_guess = [random.uniform(b[0], b[1]) for b in bounds]
    ing_loop.append(initial_guess)
    result1.append(minimize(objective_function, initial_guess, bounds = bounds))

# find the successful optimisation
solution_loop = []
for gg in result1:
    params_tmp = params.copy()
    solution = gg.x
    solution_loop.append(np.append(solution,gg.fun))
solution_table = pd.DataFrame(solution_loop,columns=['am','ap','p01','p02','tau1','n01','n02','functionvalue'])
indx = solution_table[solution_table.functionvalue<0.1].index
solution_real_table = solution_table[solution_table.functionvalue<0.1]
indx = list(indx)
result_plot = [result1[i] for i in indx]


ii=0
period_loop = []
for gg in result_plot:    
    params_tmp = params.copy()
    solution = gg.x
    # print(gg.x)
    # print(gg.fun)
    params_tmp['a_m'], params_tmp['a_p'], params_tmp['p01'],params_tmp['p02'],params_tmp['tau1'],params_tmp['n01'],params_tmp['n02'] = tuple(solution)
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    peak,_ = find_peaks(P_hes[t>T*0.8])
    period_loop.append(np.mean(np.diff(t[t>0.8*T][peak])))
    # solution_plot(params_tmp,P_hes,p_NGN,M_hes,m_NGN,y_range=[0,np.max(np.vstack((p_NGN[t>T*0.8],P_hes[t>T*0.8])))+600],peakplot=True)
    # plt.figure()
    # plt.title(ii)
    # ii+=1
    # plt.plot(t[t>T*0.8],p_NGN[t>T*0.8])

params_tmp = params.copy()
solution = result_plot[-4].x
params_tmp['a_m'], params_tmp['a_p'], params_tmp['p01'],params_tmp['p02'],params_tmp['tau1'],params_tmp['n01'],params_tmp['n02'] = tuple(solution)
params_tmp['tau2'] = 160
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
solution_plot(params_tmp,P_hes,p_NGN,M_hes,m_NGN,y_range=[0,np.max(np.vstack((p_NGN[t>T*0.8],P_hes[t>T*0.8])))+300],peakplot=True)
 

# Fix fold change of HES5 and optimise the p02 and n02, tau2 is independent of fold change of both genes 
def objective_function_transcription(vars):
    params_tmp = params.copy()
    params_tmp['p02'], params_tmp['n02']= vars
    M_hes,P_hes,m_NGN,p_NGN = M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    steady_hes = np.array([np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])])
    steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
    # fold change : NGN2 2.45 HES5 1.47
    # eq1 = 1.47 - steady_hes[-1]/steady_hes[-2]
    eq2 =  (2.45 - steady_ngn[-1]/steady_ngn[-2])**2
    
    return eq2

#
n02_loop = np.linspace(1,12,100)
error_loop1 = np.zeros_like(n02_loop)
for jj in range(len(n02_loop)):
    error_loop1[jj] = objective_function_transcription([500,n02_loop[jj]])
    
plt.plot(n02_loop,error_loop1)
plt.xlabel('Cooperation Coef')
plt.ylabel('|Fold change - 2.45|')
plt.grid(True)

#
p02_loop = np.linspace(500,1000,40)
error_loop2 = np.zeros_like(p02_loop)
for jj in range(len(p02_loop)):
    error_loop2[jj] = objective_function_transcription([p02_loop[jj],5])
    
plt.plot(p02_loop,error_loop2)
plt.xlabel('P02')
plt.ylabel('|Fold change - 2.45|')
plt.grid(True)

# Initial guess for the variables
# initial_guess = [500, 5]
bounds = [(500, 2000),(3,10)]
num_guesses = 8
result2 = []
ing_loop2 = []
for ii in range(num_guesses):
    initial_guess = [random.uniform(b[0], b[1]) for b in bounds]
    ing_loop2.append(initial_guess)
    result2.append(minimize(objective_function_transcription, initial_guess, bounds = bounds))
   
solution_loop2 = []
for gg in result2:
    params_tmp = params.copy()
    solution = gg.x
    solution_loop2.append(np.append(solution,gg.fun))
    print(gg.fun)
    params_tmp['p02'],params_tmp['n02'] = tuple(solution)
    M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
    solution_plot(params_tmp,P_hes,p_NGN,M_hes,m_NGN,True)
solution_table2 = pd.DataFrame(solution_loop2,columns=['p02','n02','functionvalue'])



