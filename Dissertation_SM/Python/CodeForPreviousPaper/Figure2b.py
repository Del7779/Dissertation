#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# parameter 
k0 = 4
k1 = 1
k2 = 1
k2_prime = 1
k3 = 1
k4 = 1
k5 = 0.1
k6 = 0.075
J3 = 0.3
J4 = 0.3

# function G
def G(u, v, J, K):
    numerator = 2 * u * K
    denominator = (v - u + v * J + u * K + np.sqrt((v - u + v * J + u * K)**2 - 4 * (v - u) * u * K))
    return numerator / denominator

def model(X,R,S):
    dRdt = k0 * G(k3*R,k4,J3,J4) + k1 * S - k2 * R - k2_prime * X * R
    dXdt = k5 * R - k6 * X
    return [dRdt, dXdt]

def Euler_method(init=[0.1,0.2],dt=0.01,tm=100,S=0.3):
    
    # Set up
    times = np.arange(0,tm,dt)
    X_value = np.zeros_like(times)
    R_value = np.zeros_like(times)
    X_value[0],R_value[0] = init
    
    for ii in range(1,len(times)):
        
        drdt,dxdt = model(X_value[ii-1],R_value[ii-1],S)
        
        X_value[ii] = dt*dxdt + X_value[ii-1]
        R_value[ii] = dt*drdt + R_value[ii-1]
        
    return (X_value,R_value,times)

S_loop = np.linspace(0,0.5,100)
Xss_loop = np.zeros_like(S_loop)
Rss_loop = np.zeros_like(S_loop)

for ii in range(len(S_loop)):
    t = 2000
    X_value,R_value, times = Euler_method(S=S_loop[ii],tm=t)
    R_range = [max(R_value[times>t*0.8]),min(R_value[times>t*0.8])]
    plt.figure(1)
    plt.plot(np.repeat(S_loop[ii],2),R_range,'b.')
    plt.xlabel('S')
    plt.ylabel('R')
    

    
# Plotting
X_values,R_values,times = Euler_method(S=0.3,tm=300)
plt.figure(figsize=(8, 6))
plt.plot(times, X_values, label='X', color='blue')
plt.plot(times, R_values, label='R', color='red')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()
    
# Phase space 
X_values,R_values,times = Euler_method(S=0.2,tm=1000)
plt.plot(X_values[times>500],R_values[times>500],'.')
Xv = np.linspace(0,1.5,100)

def equation(vars,X,S):
    R = vars
    dRdt = k0 * G(k3*R,k4,J3,J4) + k1 * S - k2 * R - k2_prime * X * R
    return dRdt

Rsolution = []
for jj in range(len(Xv)):
    
    initc = np.linspace(0,3,100)
    sol = fsolve(equation,[initc],args=(Xv[jj],0.2))
    solf = sol[np.argmin(abs(equation(sol,Xv[jj],0.2)))]
    Rsolution.append(solf)
    
plt.plot(Xv,Rsolution,'-')

Rv = np.linspace(0,3,100)
Xsolution = k5/k6 * Rv
plt.plot(Xsolution,Rv,'-')
plt.xlabel('X')
plt.ylabel('R')
plt.xlim([0,1.5])


    


        
    
        