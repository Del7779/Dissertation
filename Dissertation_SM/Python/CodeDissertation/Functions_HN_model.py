#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:03:54 2024

@author: del
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve, minimize

## contain some elementary function 

# basal degradation rates 
mu_m = np.log(2)/30
mu_p = np.log(2)/90
mu_pn= np.log(2)/40
mu_mn = np.log(2)/46

# Repression Hill function
def G1(P_tau,n,p0):
    return  1/ (1 + (P_tau / p0)**n)

# derivative
def G1_derivative(P_tau,n,p0):
    return -n * p0**n * P_tau**(n-1) / (p0**n + P_tau**n)**2

# Activation Hill function 
def G2(P_tau, n, p0):
    return P_tau**n / (p0**n + P_tau**n)

# function for crosscorrelation
def cross_correlation(x, y, max_lag,lag_shift = 10):
    lags = np.arange(-max_lag, max_lag, lag_shift)
    cross_corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            cross_corr[i] = np.corrcoef(x[:lag], y[-lag:])[0,1]
        elif lag > 0:
            cross_corr[i] = np.corrcoef(x[lag:], y[:-lag])[0,1]
        else:
            cross_corr[i] = np.corrcoef(x, y)[0,1]

    return lags, cross_corr, lag_shift

# equations for steady solutions
def model(vars,params):
    t = params.get('t')
    tau1 = params.get('tau1')
    tau2 = params.get('tau2')
    p01 = params.get('p01')
    p02 = params.get('p02')
    SS = params.get('SS', 1)
    activation = params.get('activation', False)
    a_m, a_p, a_mn, a_pn= params.get('a_m'),params.get('a_p'),params.get('a_mn'),params.get('a_pn')
    n01, n02 = params.get('n01'), params.get('n02')
    
    
    m1,p1,m2,p2 = vars
    dm1dt = a_m*G1(p1,n01,p01) - mu_m*m1
    dp1dt = a_p*m1 - mu_p*p1
    dm2dt = a_mn*G1(p1,n02,p02) - mu_mn*m2
    dp2dt = a_pn*m2 - mu_pn*p2
    return [dm1dt,dp1dt,dm2dt,dp2dt]

# plot function
def solution_plot(params, P_hes, p_NGN, M_hes, m_NGN, y_range = None, aplot=True, peakplot = False,figname=None):
    
    t = params.get('t')
    T = np.max(t)
    tau1 = params.get('tau1')
    tau2 = params.get('tau2')
    p01 = params.get('p01')
    p02 = params.get('p02')
    SS = params.get('SS', 1)
    activation = params.get('activation', False)
    a_m, a_p, a_mn, a_pn= params.get('a_m'),params.get('a_p'),params.get('a_mn'),params.get('a_pn')
    n01, n02 = params.get('n01'), params.get('n02')
    
    # fold change
    steady_ngn = np.array([np.min(p_NGN[t>T*0.9]),np.max(p_NGN[t>T*0.9])])
    fn = steady_ngn[-1]/steady_ngn[-2]
    steady_hes = np.array([np.min(P_hes[t>T*0.9]),np.max(P_hes[t>T*0.9])])
    fh = steady_hes[-1]/steady_hes[-2]
    p1,_ = find_peaks(P_hes)
    p2,_ = find_peaks(p_NGN)
    try:
        period1 = abs((t[p1][-3]-t[p1][-2])/60)
        period2 = abs((t[p2][-3]-t[p2][-2])/60)
    except Exception:
        period1 = 0
        period2 = 0
    
    if (fh - 1) <1e-2:
        period1 = 0
        period2 = 0
    # Plots
    plt.figure()
    # plt.plot(t/60, M_hes, label='mRNA(t)', color='blue')
    plt.plot(t/60, P_hes, label=f'HES5 Protein(Fold Change: {fh:.2f}, period: {period1:.2f} hr)', color='red')
    plt.plot(t/60, p_NGN, label=f'NGN2 Protein(Fold Change: {fn:.2f}), period: {period2:.2f} hr)', color='blue')

    if aplot == True:
        # initial guess 
        index = t>max(t)*0.8
        initial_guess = [np.mean(M_hes[index]),np.mean(P_hes[index]),np.mean(m_NGN[index]),
                         np.mean(p_NGN[index])]
        solutions = fsolve(lambda vars: model(vars,params),initial_guess)
        plt.axhline(solutions[1],linestyle='--',label='Analytical solution(HES5)',color='r')
        plt.axhline(solutions[3],linestyle='--',label='Analytical solution(NGN2)',color='b')
    # plt.plot(t/60, SS, label='SS', color='blue',alpha=0.6)
    plt.xlabel('Time (hrs)',fontweight='bold')
    # plt.ylim([0,3000])
    plt.ylabel('Concentration',fontweight='bold')
    if peakplot == True:
        plt.axvline(t[p1][-1]/60,linestyle='-.',color='r',label='Peak location (Hes5)')
        plt.axvline(t[p2][-1]/60,linestyle='-.',color='b',label='Peak location (Ngn2)')
    if period1 == 0:
        plt.title('No offset')
    else:
        plt.title(f'Peak offset:{np.min(np.abs([t[p1][-3]-t[p2][-3],t[p1][-3]-t[p2][-4],t[p1][-3]-t[p2][-2]]))/60:.2f}',fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_range)
    plt.tight_layout()
    # plt.show()
    if figname != None:
        plt.savefig(f'/Users/del/Library/CloudStorage/OneDrive-UniversityofStAndrews/Msc_scripts/Figures/FigureMyPc/{figname}.png',dpi=300)

## model solver
    
def minimal_model_solver(params):
    t = params.get('t')
    dt = t[1] - t[0]
    tau1 = params.get('tau1')
    tau2 = params.get('tau2')
    p01 = params.get('p01')
    p02 = params.get('p02')
    SS = params.get('SS', 1)
    activation = params.get('activation', False)
    a_m, a_p, a_mn, a_pn= params.get('a_m'),params.get('a_p'),params.get('a_mn'),params.get('a_pn')
    n01, n02 = params.get('n01'), params.get('n02')
    
    # Initial arrays for M(t) and P(t)
    M_hes = np.zeros(len(t))
    P_hes = np.zeros(len(t))
    p_NGN = np.zeros(len(t))
    m_NGN = np.zeros(len(t))

    M_hes[0] = 10 # 10
    P_hes[0] = p01  # = p0
    p_NGN[0] = p02
    m_NGN[0] = 10 # 10
    
    if activation == False:
        G = G1
    else:
        G = G2
    
    for i in range(1, len(t)):
        current_time = t[i]
        if current_time < tau1:
            past_protein = P_hes[0]
        else:
            past_protein = P_hes[i - int(tau1/dt)]
        
        M_hes[i] = M_hes[i - 1] + dt * (a_m * G(past_protein,n=n01,p0=p01) - mu_m * M_hes[i-1])
        P_hes[i] = P_hes[i - 1] + dt * (a_p * M_hes[i-1] - mu_p * SS * P_hes[i-1])
        
        # introduce second delay 
        if current_time < tau2:
            past_protein2 = P_hes[0]
        else:
            past_protein2 = P_hes[i - int(tau2/dt)]
        m_NGN[i] = m_NGN[i - 1] + dt *(SS * a_mn * G(past_protein2,n=n02,p0=p02) - m_NGN[i-1] * mu_mn)
        p_NGN[i] = p_NGN[i - 1] + dt* (m_NGN[i - 1] * a_pn - mu_pn * p_NGN[i - 1])
        
    return M_hes,P_hes,m_NGN,p_NGN 


