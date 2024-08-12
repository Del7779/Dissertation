#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters
k0 = 0
k1 = 1
k2 = 0.01
k2_prime = 10
k3 = 0.1
k4 = 0.2
k5 = 0.1
k6 = 0.05
Y_T = R_T = 1
K_m3 = K_m4 = K_m5 = K_m6 = 0.01


# ODEs
def model(X, Y_P, R_P, S,epsilon = 1e-6):
    dXdt = k0 + k1 * S - k2 * X - k2_prime * R_P * X
    dYPdt = k3 * X * (Y_T - Y_P) / max(K_m3 + Y_T - Y_P,epsilon) - k4 * Y_P / max(K_m4 + Y_P,epsilon)
    dRPdt = k5 * Y_P * (R_T - R_P) / max(K_m5 + R_T - R_P,epsilon) - k6 * R_P / max(K_m6 + R_P,epsilon)
    return dXdt, dYPdt, dRPdt

def Euler_method(S,dt=0.01,tm=100,X0=5,YP0=0.9,RP0=0.1):
    
    # Time parameters, update N times
    times = np.arange(0,tm,dt)

    # Arrays to store the solutions
    X_values = np.zeros(len(times))
    YP_values = np.zeros(len(times))
    RP_values = np.zeros(len(times))

    # Initial conditions according to the plot in paper
    X_values[0] = 5
    YP_values[0] = 0.9
    RP_values[0] = 0.1
    
    # Euler integration
    for ii in range(1, len(times)):
        
        # S=2      
        dXdt, dYPdt, dRPdt = model(X_values[ii-1], YP_values[ii-1], RP_values[ii-1], S)
        X_values[ii] = dt*dXdt + X_values[ii-1]
        YP_values[ii] = dt*dYPdt + YP_values[ii-1]
        RP_values[ii] = dt*dRPdt + RP_values[ii-1]
        # check 
        # Visual check 
        #if ii % 20 == 0:
            #print("check every 20 steps:")
            #print(f'R{dRPdt}\nX{dXdt}\nYP{dYPdt}')
        
        #if np.any(np.isnan(np.array([X,Y_P,R_P]))):
            #print(X, Y_P, R_P,"/")
            #print(dXdt,dYPdt,dRPdt)
            #break
            
    return (X_values,YP_values,RP_values,times)

    
# Signal-Response 
S_loop = np.linspace(0.1,7,100)
Xss = np.zeros_like(S_loop)
YPss = np.zeros_like(S_loop)
Rpss_a = np.zeros_like(S_loop)
RPss = np.zeros_like(S_loop)

# Analytical solution 
def equation(vars,S):
    epsilon = 1e-6
    Y_P,R_P,X = vars
    eq1 = k0 + k1 * S - k2 * X - k2_prime * R_P * X
    eq2 = k3 * X * (Y_T - Y_P) / max(K_m3 + Y_T - Y_P,epsilon) - k4 * Y_P / max(K_m4 + Y_P,epsilon)
    eq3 = k5 * Y_P * (R_T - R_P) / max(K_m5 + R_T - R_P,epsilon) - k6 * R_P / max(K_m6 + R_P,epsilon)
    return [eq1,eq2,eq3]

for i in range(len(S_loop)):
    
    # when s is close to Hopf point, it takes longer steps to relax
    if S_loop[i] < 5.5:
        t = 1000
        X_values,YP_values,RP_values,times = Euler_method(S_loop[i],tm=t)
    else:
        t = 10000
        X_values,YP_values,RP_values,times = Euler_method(S_loop[i],tm=t)
        
    RP_range = [max(RP_values[times>t*0.8]),min(RP_values[times>t*0.8])]
    plt.figure(1)
    plt.plot(np.repeat(S_loop[i],len(RP_range)),RP_range,'b.')
    
    
    # Solve the system of equations
    solution = fsolve(equation,[0.5,0.5,3],args = (S_loop[i]))
    Rpss_a[i] = solution[1]

plt.figure(1)
plt.plot(S_loop,Rpss_a,'r.')
plt.xlabel('S')
plt.ylabel('R')    
            
# Plotting
X_values,YP_values,RP_values,times = Euler_method(3,tm=60)
plt.figure(figsize=(12, 6))
plt.plot(times, X_values, label='X', color='blue')
plt.plot(times, YP_values*5, label='Y_P', color='red')
plt.plot(times, RP_values*5, label='R_P', color='green')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Negative-feedback Oscillator Dynamics')
plt.ylim([0,8])
plt.legend()
plt.show()

