#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:39:32 2024

@author: Yunhao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the model parameters
#alpha_m = 0.5  # Production rate of M
mu_m = 0.03    # Degradation rate of M
#alpha_p = 0.2  # Production rate of P
mu_p = 0.03   # Degradation rate of P
#tau = 20    # Delay in seconds
P0 = 100      # Half-maximal effect concentration of P
n = 5        # Hill coefficient
T = 800        # Total time of simulation
dt = 0.01     # Time step

# time
t = np.arange(0, T + dt, dt)

# Initial arrays for M(t) and P(t)
M = np.zeros(len(t))
P = np.zeros(len(t))
M[0] = 3
P[0] = 100

# parameter of distribution of tau (uniform)
s = 5
mtau = 18.5
nsam = 1000
# Function to calculate G based on delayed P
def G(P_tau):
    return (1 / (1 + (P_tau / P0)**n))

#tau = np.linspace(mtau-s,mtau+s,nsam)
#dtau = tau[2] - tau[1]

tau = 18.5
integer_delay = tau/dt
# Euler integration with delay      
for i in range(1, len(t)):
    
    # You can use the previous exact solution
    #ptau = np.where(tau>t[i],P[-1],P[np.round((t[i] - tau) % dt.astype(int)])
    
    # or you can extrapolate 
    # Define an interpolation function for past p values
    p_interp = interp1d(t[:i+1], P[:i+1], kind='linear', fill_value="extrapolate")
    
    #ptau = np.where(tau>t[i],P[-1],p_interp(t[i]-tau))
    #print(ptau)
    #print(ptau.shape)
    current_time = t[i]
    if current_time < tau:
        past_protein = 100
    else:
        past_protein = P[i - int(integer_delay)]
        #print('the protein at time .. is ..')
        #print(current_time)
        #print(past_protein)
    M[i] = M[i - 1] + dt * (G(past_protein) - mu_m * M[i - 1])
    P[i] = P[i - 1] + dt * (M[i - 1] - mu_p * P[i - 1])

# Plots
plt.figure(figsize=(10, 5))
plt.plot(t, M, label='mRNA(t)', color='blue')
plt.plot(t, P, label='Protein(t)', color='red')
plt.title('Hes1 expression dynamics')
plt.xlabel('Time')
plt.ylabel('Scaled Copy Number')
plt.legend()
plt.grid(True)
# plt.ylim([0,12])
plt.show()
