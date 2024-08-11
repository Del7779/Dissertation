#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Functions_HN_model import *


params = {
    't': np.arange(0, 10000 + 0.09, 0.09),  # Example time array
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
dt = t[2] - t [1]

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

# correlation analysis tau1 
tau1_loop = np.linspace(40,100,50)
period_loop = np.zeros((2,len(tau1_loop)))
lagpeak_loop = np.zeros_like(tau1_loop)
corm_loop = np.zeros_like(tau1_loop)
cor0_loop = np.zeros_like(tau1_loop)

for jj in range(len(tau1_loop)):
    
    params_tmp = params.copy()
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

# The fold change of NGN2 is independet of a_mn and a_np
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params)
# initial guess 
index = t>max(t)*0.8
initial_guess = [np.mean(M_hes[index]),np.mean(P_hes[index]),np.mean(m_NGN[index]),
                 np.mean(p_NGN[index])]
solutions = fsolve(lambda vars: model(vars,params),initial_guess)
model(solutions,params)

K = G1_derivative(solutions[1],5,params['p01'])
a_m = params['a_m']
a_p = params['a_p']
# Define the system of equations
def equations(vars, z):
    y, x = vars
    eq1 = x**2 - y**2 + (mu_m + mu_p)*x + mu_m*mu_p - a_m*a_p*K*np.cos(y*z)
    eq2 = 2*x*y + (mu_m + mu_p)*y + a_m*a_p*K*np.sin(y*z)
    return [eq1, eq2]

# Define a range for z
z_values = np.linspace(40, 200, 30)
y_solutions1 = [] 

# Initial guess for [y, x]
initial_guess_y = 2*np.pi/4
C = (mu_m * mu_p - a_m * a_p * K * np.cos(initial_guess_y * z_values[0]) - initial_guess_y**2)
B = (mu_m + mu_p)
X1_pos = (-B + np.sqrt(B**2 - 4 * C)) / 2
initial_guess = [initial_guess_y, X1_pos]

# Solve the equations for each z
for z in z_values:
    solution = fsolve(equations, initial_guess, args=(z))
    y_solutions1.append(solution[0])  # Append only the y value
    initial_guess[0] = solution[0]  # Update the initial guess for better convergence
    C = (mu_m * mu_p - a_m * a_p * K * np.cos(solution[0] * z) - solution[0]**2)
    B = (mu_m + mu_p)
    X1_pos = (-B + np.sqrt(B**2 - 4 * C)) / 2
    initial_guess[1] = X1_pos
    # initial_guess = solution  # Update the initial guess for better convergence

# Plotting
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
plt.figure()
plt.plot(z_values, (2*np.pi)/np.array(y_solutions1)/60, label='Analytical',linewidth=3)
plt.plot(tau1_loop,period_loop[0,:],label = 'Numerical',linewidth=3)
plt.xlabel(r'$\tau$/hrs')
plt.ylabel(r'Period/hrs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim([42,60])
plt.ylim([4,6.5])
plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/tauperiodanalytical.png',dpi=300)
plt.show()



y_solutions2 = []    
# Initial guess for [y, x]
initial_guess_y = 2
C = (mu_m * mu_p - a_m * a_p * K * np.cos(initial_guess_y * z_values[0]) - initial_guess_y**2)
B = (mu_m + mu_p)
X1_neg = (-B - np.sqrt(B**2 - 4 * C)) / 2
initial_guess = [initial_guess_y, X1_neg]

# Solve the equations for each z
for z in z_values:
    solution = fsolve(equations, initial_guess, args=(z))
    y_solutions2.append(solution[0])  # Append only the y value
    initial_guess[0] = solution[0]  # Update the initial guess for better convergence
    C = (mu_m * mu_p - a_m * a_p * K * np.cos(solution[0] * z) - solution[0]**2)
    B = (mu_m + mu_p)
    X1_neg = (-B - np.sqrt(B**2 - 4 * C)) / 2
    initial_guess[1] = X1_neg
    
    
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(z_values, (2*np.pi)/np.array(y_solutions2)/60, label='y(z)')
plt.plot(tau1_loop,period_loop[1,:])
plt.xlabel('z')
plt.ylabel('y')
plt.title('Relationship between y and z')
plt.legend()
plt.grid(True)
plt.show()
