#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:27:31 2024

@author: del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Functions_HN_model import * 

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
params = {
    't': np.arange(0, 20000 + 0.09, 0.09),  # Example time array
    'tau1': 65,
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
    'n02': 1
}
t = params['t']
T = max(t)
dt = t[2] - t[1]

## This snippet set up the optimised parameters 
# params_tmp = params.copy()
# solution = result1[37].x
# print(gg.x)
# print(gg.fun)
# params_tmp['a_m'], params_tmp['a_p'], params_tmp['p01'],params_tmp['p02'],params_tmp['tau1'],params_tmp['n01'],params_tmp['n02'] = tuple(solution)
# params = params_tmp.copy()
M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params)
# solution_plot(params,P_hes,p_NGN,M_hes,m_NGN,y_range=[0,np.max(np.vstack((p_NGN[t>T*0.8],P_hes[t>T*0.8])))+600])
# plt.xlim([min(time)/60,max(time)/60])

# Load the data
data = P_hes[t>T*0.9]
data = G1(data,params['n02'],params['p02'])
data_central = data
time = t[t>T*0.9]
values = data_central

# Perform Fourier Transform
fft_values = np.fft.fft(values)
ofrequencies = np.fft.fftfreq(len(values), d=(time[1] - time[0]))
frequencies = 2*np.pi*ofrequencies

# calculate the p_NGN according to analytical solution
t1 = np.linspace(min(time),max(time),1000) + params['tau2']
nt = np.zeros_like(t1)
for ii in range(len(t1)):
    nt[ii] = params['a_mn']*params['a_pn']*sum(np.exp(1j*frequencies*t1[ii])/(mu_mn+1j*frequencies)/(mu_pn+1j*frequencies)*fft_values)/len(values)
    # nt[ii] = np.real(sum(np.exp(1j*frequencies*t1[ii])*fft_values)/len(values))

plt.figure()
# plt.plot(time,data_central)
plt.plot(t/60,p_NGN,label = 'Simulation(Ngn2)',alpha = 0.7)
plt.plot(t1/60,nt,label = 'Solution using Fourier Transform',linestyle = '--')
# plt.plot(t/60,P_hes,alpha=0.5,label = 'Simulation(Hes5)')
plt.xlim([min(time)/60+2,max(time)/60])
plt.ylim
plt.xlabel('Time/hrs')
plt.ylabel(r'$p_2$')
plt.grid(True)
plt.tight_layout()
# plt.ylim([0,100])
plt.legend()
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/FourierAna.png',dpi=300)




# Analyze the frequency spectrum
magnitude = np.abs(fft_values)
dominant_indices = np.argsort(magnitude)[-2]  # Get indices of the top 3 frequencies
dominant_frequencies = ofrequencies[dominant_indices]

# Define the fit function
def fit_function(t, *params):
    result = params[0] * np.ones_like(t)
    # result = np.zeros_like(t)
    for i in range((len(params)-1) // 2):
        A1 = params[1 + 2 * i]
        # A2 = params[1 + 3 * i + 1]
        omega = params[1 + 2 * i + 1]
        phi = params[1 + 2 * i + 2]
        # result += A1 * np.sin(omega * t) + A2 * np.cos(omega * t) 
        result += A1 * np.cos(omega * t + phi) 
    return result

# Initial guess: Amplitudes, Frequencies, Phases
initial_guess = [np.mean(values)]
# for freq in dominant_frequencies:
initial_guess.extend([np.max(values), 2 * np.pi * dominant_frequencies,0])

# Fit the function to the data
Params, _ = curve_fit(fit_function, time, values, p0=initial_guess)

omg = Params[2]
amp = Params[1]
phi = Params[3]
# calculate the p_NGN 
# nt = params['a_mn']*(Params[0]/mu_mn+ amp/2/(mu_mn**2+omg**2)*((mu_mn-omg*1j)*np.exp(1j*omg*t1) + (mu_mn+omg*1j)*np.exp(-1j*omg*t1)))
nt = params['a_mn']*params['a_pn']*((Params[0]/mu_mn/mu_pn) + amp/np.sqrt((mu_mn**2 + omg**2)*(mu_pn**2+omg**2))*np.cos(omg*t1+np.arctan(-omg/mu_mn)+np.arctan(-omg/mu_pn)+phi))
# plt.plot(time,data_central)
plt.figure()
plt.plot(t/60,p_NGN,label = 'Simulation',alpha = 0.4)
plt.plot(t1/60,nt,label = 'Solution using the first harmonic',linestyle = '--')
# plt.ylim([0,2500])
plt.xlabel('Time/hrs')
plt.ylabel(r'$p_2$')
plt.xlim([min(time)/60+2,max(time)/60])
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/FourierFirstHarmonic.png',dpi=300)

plt.figure()
# Plot the original and fitted data
plt.plot(time, values, label='Signal')
plt.plot(time, fit_function(time, *Params), label='Fitted Function', linestyle='--')
# plt.plot(t1,(fit_function(t1, *Params)- np.mean(fit_function(t1, *Params)))/np.std(fit_function(t1, *Params)))
# plt.plot(t1,(nt-np.mean(nt))/np.std(nt))
plt.legend()
plt.xlim([min(time),max(time)])
plt.xlabel('Time')
plt.ylabel('Signal (G(HES5,5,500)')
plt.title('Original vs Fitted Data')
plt.show()

