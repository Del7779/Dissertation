#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:52:01 2024

@author: del
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import fsolve, minimize
import seaborn as sns
from Functions_HN_model import *

# Time
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


a_pn_loop = np.linspace(0.1,40,40)
a_mn_loop = np.linspace(0.1,40,40)

fold_change = np.zeros((len(a_pn_loop),len(a_mn_loop)))
for ii in range(len(a_pn_loop)):
    for jj in range(len(a_mn_loop)):
        params_tmp = params        
        params_tmp['a_pn'], params_tmp['a_mn'] = a_pn_loop[ii], a_mn_loop[jj]
        M_hes,P_hes,m_NGN,p_NGN = minimal_model_solver(params_tmp)
        steady_ngn = np.array([np.min(p_NGN[t>T*0.8]),np.max(p_NGN[t>T*0.8])])
        fold_change[ii,jj] = steady_ngn[-1]/steady_ngn[-2]
        

# Plot heat map
import pandas as pd
data = pd.DataFrame(np.round(fold_change,2), index=np.round(a_pn_loop,2), columns=np.round(a_mn_loop,2))
sns.heatmap(data, annot=False, cmap='viridis', linewidths=.5)
plt.xlabel(r'Transcription Rate $a_{m_2}$')
plt.ylabel(r'Translation Rate $a_{p_2}$')
# plt.title('Fold change of NGN2 against different combinations of am and ap')
# Show the plot
plt.tight_layout()
plt.savefig('amap_foldchange.png', dpi=300)
plt.show()
