#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

## HES5 auto-regulation

# Time
n = 5        # Hill coefficient
T = 6000       # Total time of simulation
dt = 0.01     # Time step
t = np.arange(0, T + dt, dt)

# Initial arrays for M(t) and P(t)
M_hes = np.zeros(len(t))
P_hes = np.zeros(len(t))
p_NGN = np.zeros(len(t))
m_NGN = np.zeros(len(t))

P0 = 3400 # 34000
M_hes[0] = 100
P_hes[0] = P0  # = p0
p_NGN[0] = P0
m_NGN[0] = 100



# basal producation rates
a_m = 1000
a_p = 1000
a_pn = 10
a_mn = 100

# basal degradation rates 
mu_m = np.log(2)/30
mu_p = np.log(2)/90
mu_pn= np.log(2)/40
mu_mn = np.log(2)/46

# signal(NGN2) 
S_m = 8
S_p = 1
SS = 8


# Function to calculate G based on delayed P
def G(P_tau,n,p0):
    return (1 / (1 + (P_tau / p0)**n))

tau = 40
tau2 = np.linspace(10,120,30)
tau2_tmp = 120
tau3 = 90 # for hes5 autorepression
n0 = 3

# Full model 
# Euler integration with delay   
for i in range(1, len(t)):
    
    # Hes autorepression
    current_time = t[i]
    if current_time < tau:
        past_protein = P_hes[-1]
    else:
        past_protein = P_hes[i - int(tau/dt)]
    
    M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=P0) - mu_m * M_hes[i-1])
    P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * SS * P_hes[i-1])
    # delay ?
    if current_time < tau2_tmp:
        past_protein2 = P_hes[-1]
    else:
        past_protein2 = P_hes[i - int(tau2_tmp/dt)]
    
    
    # Ngn2 autorepression
    if current_time < tau3:
        past_protein3 = p_NGN[-1]
    else:
        past_protein3 = p_NGN[i - int(tau3/dt)]
    
    
    m_NGN[i] = m_NGN[i - 1] + dt *(SS * a_mn * G(past_protein2,n=n0,p0=P0) * G(past_protein3,n=n0,p0=P0) - m_NGN[i-1] * mu_mn)
    p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])

# Plots
plt.figure(1,figsize=(10, 5))
plt.plot(t, M_hes/np.mean(M_hes[t>T*0.8]), label='mRNA(t)', color='blue')
plt.plot(t, P_hes/np.mean(P_hes[t>T*0.8]), label='Protein(t)', color='red')
plt.plot(t, p_NGN/np.mean(p_NGN[t>T*0.8]), label='NGN2_Protein(t)', color='black')
plt.axhline(1,color = [0.2,0.9,0.2],linestyle='--')
plt.title(f'Hes5 expression dynamics (tau = {[tau,tau2_tmp,tau3]},n={n0},p={P0})')
plt.xlabel('Time')
plt.ylabel('Scaled Copy Number')
plt.legend()
plt.ylim([0,10])
# plt.xlim([1000,1500])
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


# # Find peaks
# plt.figure(2)
# peaks, _ = find_peaks(p_NGN)
# plt.plot(np.diff(t[peaks]/60),'go',label='peaks')
# # Find valleys by inverting the data
# valleys, _ = find_peaks(-p_NGN)
# plt.plot(np.diff(t[valleys])/60,'ro',label='vallyes')
# plt.xlabel('Period')
# plt.legend()

def cross_correlation(x, y, max_lag):
    lags = np.arange(-max_lag, max_lag + 1,3)
    cross_corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            cross_corr[i] = np.corrcoef(x[:lag], y[-lag:])[0,1]
        elif lag > 0:
            cross_corr[i] = np.corrcoef(x[lag:], y[:-lag])[0,1]
        else:
            cross_corr[i] = np.corrcoef(x, y)[0,1]

    return lags, cross_corr

def cc_plot(x1,x2,t,tlow,tup,dt,maxlags=None):
    plt.figure
    index = (t < tup) & (t>tlow)
    if maxlgas == None:       
        lags, cc = cross_correlation(x1[index],x2[index],int(len(index)/2))
    else:
        lags, cc = cross_correlation(x1[index],x2[index],maxlags)
    plt.plot(lags*dt/60,cc,'.',markersize=0.2)
    plt.grid(True)

