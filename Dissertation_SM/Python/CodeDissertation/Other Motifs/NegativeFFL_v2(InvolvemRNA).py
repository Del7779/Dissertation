#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from Functions_HN_model import * 

# Time
n = 5        # Hill coefficient
T = 4000       # Total time of simulation
dt = 0.01     # Time step
t = np.arange(0, T + dt, dt) # unit : hrs  

# Function to calculate G based on delayed P
def G(P_tau,n,p0,b0=0,b1=1):
    return  b0 + ((b1 - b0) / (1 + (P_tau / p0)**n))

# bounded sigmoid function to control signal 
def bounded_sigmoid(x, b0, b1, k=1, x0=0):
    return b0 + (b1 - b0) / (1 + np.exp(-k * (x - x0)))

# function for crosscorrelation
def cross_correlation(x, y, max_lag):
    lags = np.arange(-max_lag, max_lag + 1,4)
    cross_corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            cross_corr[i] = np.corrcoef(x[:lag], y[-lag:])[0,1]
        elif lag > 0:
            cross_corr[i] = np.corrcoef(x[lag:], y[:-lag])[0,1]
        else:
            cross_corr[i] = np.corrcoef(x, y)[0,1]

    return lags, cross_corr

# Cross-correlation plot
def cc_plot(x1,x2,tindex,dt,maxlags=None):
    plt.figure
    if maxlags == None:       
        lags, cc = cross_correlation(x1[tindex],x2[tindex],int(sum(tindex)/2))
    else:
        lags, cc = cross_correlation(x1[tindex],x2[tindex],maxlags)
    plt.plot(lags*dt/60,cc,'.',markersize=0.2)
    plt.grid(True)

def model_solver(tau1,tau2,SS=1,p01,p02):
    # Initial arrays for M(t) and P(t)
    M_hes = np.zeros(len(t))
    P_hes = np.zeros(len(t))
    p_NGN = np.zeros(len(t))
    m_NGN = np.zeros(len(t))

    M_hes[0] = 10
    P_hes[0] = p01  # = p0
    p_NGN[0] = p02
    m_NGN[0] = 10

    for i in range(1, len(t)):
        current_time = t[i]
        if current_time < tau1:
            past_protein = P_hes[0]
        else:
            past_protein = P_hes[i - int(tau1/dt)]
        
        M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=p01) - mu_m * M_hes[i-1])
        P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * SS * P_hes[i-1])
        # delay ?
        if current_time < tau2:
            past_protein2 = P_hes[0]
        else:
            past_protein2 = P_hes[i - int(tau2/dt)]
        m_NGN[i] = m_NGN[i - 1] + dt *(SS * a_mn * G(past_protein2,n=n0,p0=p02) - m_NGN[i-1] * mu_mn)
        p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])
        
    return M_hes,P_hes,m_NGN,p_NGN 


def model_solver2(tau1,tau2,SS,p01,p02):
    # Initial
    M_hes = np.zeros(len(t))
    P_hes = np.zeros(len(t))
    p_NGN = np.zeros(len(t))
    m_NGN = np.zeros(len(t))
    M_hes[0] = 10
    P_hes[0] = p01  # = p0
    p_NGN[0] = p02
    m_NGN[0] = 10
    
    if SS == None:     
        SS = np.zeros_like(P_hes)
        for i in range(1, len(t)):
            current_time = t[i]
            if current_time < tau1:
                past_protein = P_hes[0]
            else:
                past_protein = P_hes[i - int(tau1/dt)]
                
            SS[i] = SS[i-1] + dt*(a_m*p_NGN[i-1] - 10*SS[i-1])
            M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=p01) - mu_m * M_hes[i-1])
            P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * G(SS[i-1],3,25,b0=0,b1=70) * P_hes[i-1])
            
            if current_time < tau2:
                past_protein2 = P_hes[0]
            else:
                past_protein2 = P_hes[i - int(tau2/dt)]
            m_NGN[i] = m_NGN[i - 1] + dt *(G(SS[i-1],3,25,b0=0,b1=70) * a_mn * G(past_protein2,n=n0,p0=p02) - m_NGN[i-1] * mu_mn)
            p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])
    else:
        
        for i in range(1, len(t)):
            current_time = t[i]
            if current_time < tau1:
                past_protein = P_hes[0]
            else:
                past_protein = P_hes[i - int(tau1/dt)]
            M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=p01) - mu_m * M_hes[i-1])
            P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * G(SS,3,25,b0=0,b1=70) * P_hes[i-1])
            # delay ?
            if current_time < tau2:
                past_protein2 = P_hes[0]
            else:
                past_protein2 = P_hes[i - int(tau2/dt)]
            m_NGN[i] = m_NGN[i - 1] + dt *(G(SS,3,25,b0=0,b1=70) * a_mn * G(past_protein2,n=n0,p0=p02) - m_NGN[i-1] * mu_mn)
            p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])
  
    return M_hes,P_hes,m_NGN,p_NGN,SS


# Initial arrays for M(t) and P(t)
M_hes = np.zeros(len(t))
P_hes = np.zeros(len(t))
p_NGN = np.zeros(len(t))
m_NGN = np.zeros(len(t))

P01 = 300 # 34000
P02 = 500 # 
M_hes[0] = 10
P_hes[0] = P01  # = p0
p_NGN[0] = P02
m_NGN[0] = 10

# basal producation rates
a_m = 1
a_p = 10 # 10
a_pn = 4 # 4
a_mn = 1 # 1 

# basal degradation rates 
mu_m = np.log(2)/30
mu_p = np.log(2)/90
mu_pn= np.log(2)/40
mu_mn = np.log(2)/46

# signal(NGN2) 
SS = 1
SS = np.zeros_like(P_hes)

# delay
tau = 40
tau2 = np.linspace(1,120,80)
n0=5

# Full model 
# Euler integration with delay  
tau2_tmp = 25
for i in range(1, len(t)):
    current_time = t[i]
    if current_time < tau:
        past_protein = P_hes[0]
    else:
        past_protein = P_hes[i - int(tau/dt)]
    SS[i] = SS[i-1] + dt*(a_m*p_NGN[i-1] - 10*SS[i-1])
    M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=P01) - mu_m * M_hes[i-1])
    P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - G(SS[i-1],10,60,b0=0,b1=70) * mu_p * P_hes[i-1])
    # delay ?
    if current_time < tau2_tmp:
        past_protein2 = P_hes[0]
    else:
        past_protein2 = P_hes[i - int(tau2_tmp/dt)]
    m_NGN[i] = m_NGN[i - 1] + dt * (G(SS[i-1],10,60,b0=0,b1=70) * a_mn * G(past_protein2,n=n0,p0=P02) - m_NGN[i-1] * mu_mn)
    p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])

#
def solution_t():
    # Plots
    plt.figure(1,figsize=(10, 5))
    # plt.plot(t/60, M_hes, label='mRNA(t)', color='blue')
    plt.plot(t/60, P_hes, label='Protein(t)', color='red')
    plt.plot(t/60, p_NGN, label='NGN2_Protein(t)', color='black')
    # plt.plot(t/60, SS, label='SS', color='blue',alpha=0.6)
    plt.title('Hes5 expression dynamics')
    plt.xlabel('Time')
    # plt.ylim([0,3000])
    plt.ylabel('Scaled Copy Number')
    plt.legend()
    plt.grid(True)
    plt.show()


# crosscorelation plot
cc_plot(P_hes,p_NGN,(t>5*15) & (t<5*17),dt)

# P01,P02
P02_loop = np.linspace(1,2500,70)

for ii in range(len(P02_loop)):
    # # control signal(NGN2) 
    # SS = 5
    
    P02 = P02_loop[ii]
    M_hes,P_hes,m_NGN,p_NGN,SS = model_solver2(tau,25,None,P01,P02)
    steady_hes = [np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])]
    steady_ngn = [np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])]
    plt.figure(1)
    plt.plot([P02,P02],steady_hes,'r.')
    plt.xlabel('Repression Threshold P01')
    plt.ylabel('Hes Steady State')
    plt.figure(2)
    plt.xlabel('Repression Threshold P02')
    plt.ylabel('NGN2 Steady State')
    plt.plot([P02,P02],steady_ngn,'b.')
    

    
# SS
plt.figure
SS_loop = np.linspace(0,100,100)
# plt.plot(SS_loop,G(SS_loop,2,p0=25,b0=0,b1=80))

for ii in range(len(SS_loop)):
    P01 = 300 # 34000
    P02 = 250
    M_hes,P_hes,m_NGN,p_NGN = model_solver(tau, 25, SS_loop[ii], P01, P02)
    steady_hes = [np.min(P_hes[t>T*0.8]),np.max(P_hes[t>T*0.8])]
    steady_ngn = [np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])]
    plt.figure(1)
    plt.plot([SS_loop[ii],SS_loop[ii]],steady_hes,'r.',label='HES5')
    plt.xlabel('SS')
    plt.ylabel('HES5 steady state')
    plt.figure(2)
    plt.xlabel('SS')
    plt.ylabel('NGN2 steady state')
    plt.plot([SS_loop[ii],SS_loop[ii]],steady_ngn,'b.')
  
    
    
    
tau2 = np.linspace(2,120,80)
tau1 = np.linspace(2,120,80)
period_loop = np.zeros((2,len(tau2)))
lagpeak_loop = np.zeros_like(tau2)
corm_loop = np.zeros_like(tau2)
SS=1

for jj in range(len(tau2)):
    
    P01 = 300 
    P02 = 250
    tau2_tmp = tau2[jj]
    
    M_hes,P_hes,m_NGN,p_NGN = model_solver(40, tau2_tmp, SS, P01, P02)
    # for i in range(1, len(t)):
    #     current_time = t[i]
    #     if current_time < tau:
    #         past_protein = P_hes[0]
    #     else:
    #         past_protein = P_hes[i - int(tau/dt)]
        
    #     M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n0,p0=P01) - mu_m * M_hes[i-1])
    #     P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * SS * P_hes[i-1])
    #     # 
    #     if current_time < tau2_tmp:
    #         past_protein2 = P_hes[0]
    #     else:
    #         past_protein2 = P_hes[i - int(tau2_tmp/dt)]
    #     m_NGN[i] = m_NGN[i - 1] + dt *(SS * a_mn * G(past_protein2,n=n0,p0=P02) - m_NGN[i-1] * mu_mn)
    #     p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])
        
    peaks, _ = find_peaks(P_hes)
    period1 = np.mean(np.diff(t[peaks[-3:-1]]))
    
    period_loop[0,jj] = period1/60
    
    peaks, _ = find_peaks(p_NGN)
    period2 = np.mean(np.diff(t[peaks[-3:-1]]))
    
    period_loop[1,jj] = period2/60
    
    period = (period1+period2)/2
    
    # index = (t>period*15) & (t<period*17)
    # lags, cc = cross_correlation(p_NGN[index],P_hes[index],int(sum(index)/2))
    lags, cc = cross_correlation(p_NGN[-int(2*period/dt):-1],P_hes[-int(2*period/dt):-1],int(2*period/dt/2))
    lags = lags*dt
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
    
    



plt.figure(3)
plt.plot(lags/60,cc)
plt.xlabel('lag')
plt.ylabel('Pearson Correlation')
plt.grid(True)

plt.figure(4)
p_lag_ind = (np.sign(corm_loop)>0)
n_lag_ind = (np.sign(corm_loop)<0)
plt.plot(tau2[p_lag_ind]/60,lagpeak_loop[p_lag_ind],'r.',label='Positive Correlation')
plt.plot(tau2[n_lag_ind]/60,lagpeak_loop[n_lag_ind],'b.',label='Negative Correlation')
plt.ylabel('the lag of the strongest correlation (hr)')
plt.xlabel('tau2 (hr)')
plt.legend()

plt.figure
plt.plot(tau2/60,corm_loop)
plt.xlabel('tau')
plt.ylabel('Maximum Pearson Correlation')

plt.figure(1)
plt.plot(tau2,period_loop[,:],'.')
plt.xlabel('Tau')
plt.ylabel('Period(hrs)')

    
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



