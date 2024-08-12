#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

## HES5 auto-regulation

# Time
n = 5        # Hill coefficient
T = 2000       # Total time of simulation
dt = 0.01     # Time step
t = np.arange(0, T + dt, dt)

# Initial arrays for M(t) and P(t)
M_hes = np.zeros(len(t))
P_hes = np.zeros(len(t))
p_NGN = np.zeros(len(t))

P0 = 1000 # 34000
M_hes[0] = 500
P_hes[0] = P0  # = p0
p_NGN[0] = 500


# basal producation rates
a_m = 10
a_p = 10
a_pn = 10

# basal degradation rates 
mu_m = np.log(2)/30
mu_p = np.log(2)/90
mu_pn= np.log(2)/40

# signal(NGN2) 
S_m = 8
S_p = 1
SS = 4
 
# parameter of distribution of tau (uniform)
# taumin = 5
# taumax = 40
# nsam = 1000
# tau = np.linspace(taumin,taumax,nsam)
# dtau = tau[2] - tau[1]


# Function to calculate G based on delayed P
def G(P_tau,n,p0):
    return (1 / (1 + (P_tau / p0)**n))

tau = 40
tau2 = np.linspace(10,120,30)

# Full model 
# Euler integration with delay   
   
tau2_tmp = 40
for i in range(1, len(t)):
    #p_interp = interp1d(t[:i+1], P[:i+1], kind='linear', fill_value="extrapolate")
    current_time = t[i]
    if current_time < tau:
        past_protein = P_hes[-1]
    else:
        past_protein = P_hes[i - int(tau/dt)]
    
    M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=5,p0=P0) - mu_m * M_hes[i-1])
    P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * SS * P_hes[i-1])
    # delay ?
    if current_time < tau2_tmp:
        past_protein2 = P_hes[-1]
    else:
        past_protein2 = P_hes[i - int(tau2_tmp/dt)]
    
    
    p_NGN[i] = p_NGN[i - 1] + dt *(SS * a_pn * G(past_protein2,n=5,p0=P0) - p_NGN[i-1] * mu_pn)
    # p_NGN[i] = p_NGN[i - 1] + dt *(SS * a_pn - p_NGN[i-1] * mu_pn * P_hes[i - 1] - p_NGN[i-1] * mu_pn)
    # print(p_NGN[i],P_hes[i])

    
    if np.isnan(p_NGN[i]):
        break
       
# Plots
plt.figure(1,figsize=(10, 5))
plt.plot(t/60, P_hes, label='Protein(t)', color='red')
plt.plot(t/60, p_NGN, label='NGN2_Protein(t)', color='black')
plt.title('Hes5/NGN2 expression dynamics')
plt.xlabel('Time(hour)')
plt.ylabel('Scaled Copy Number')
plt.legend()
# plt.ylim([0,10])
plt.grid(True)
    

    # plt.figure(1)
    # plt.plot(np.repeat(tau2_tmp,2),[min(p_NGN[t>T*0.8]),max(p_NGN[t>T*0.8])],'b.')
    


# # Hill repression function
# pp = np.linspace(0,10000,100)
# plt.figure(3)
# n=5
# for ii in [100,300,500,1000,3400]:
#     plt.axvline(x=ii, color=[ii/3400,0,0], linestyle='--')
#     plt.plot(pp,G(pp,n,ii),color = [ii/3400,0,0],label=f'P0={ii}')
#     plt.title(f'n = {n}')
# plt.xlim([0,6000])
# plt.legend()



