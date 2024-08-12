#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read time series 
t_hes5 = pd.read_csv('data/2022_07_05_p2p4_HES5.csv').to_numpy()
t_ngn2 = pd.read_csv('data/2022_07_05_p2p4_NGN2.csv').to_numpy()
t_hes5.shape

# moving average 
def moving_average(array,n=3):
    array_ma = np.zeros_like(array)
    for ii in range(len(array)):
        array_ma[ii] = np.mean(array[np.max([0,ii-n]):np.min([len(array)-1,ii+n])])   
    return array_ma

def cross_correlation(x, y, max_lag):
    """
    Calculate the cross-correlation of two time series with specified lags.

    Parameters:
    x (array-like): First time series.
    y (array-like): Second time series.
    max_lag (int): Maximum lag to calculate cross-correlation for (both positive and negative).

    Returns:
    lags (numpy array): Array of lags.
    cross_corr (numpy array): Cross-correlation values for each lag.
    """
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

plt.figure(3)
t_dt = np.mean(np.diff(t_hes5[:,0]))


c1 = [87, 3, 91, 58, 29, 28, 57, 45, 68, 21, 16, 14, 12, 27, 49, 23, 92, 88, 9]
c2 = [90, 39, 93, 46, 78, 15, 82, 48, 2, 37, 65, 10, 77, 64, 5]
c3 = [69, 33, 71, 38, 80, 61, 13, 41, 31, 17]
c4 = [30, 6, 79, 56, 85, 53, 40, 26, 75]
c5 = [67, 43, 86, 8, 34]
c6 = [70, 7, 24, 59, 36, 42, 11, 66, 89, 83, 72, 32, 44]
c_loop = (c1,c2,c3,c4,c5,c6)

plt.figure(1)
for jj in c_loop:
    t_hes_a = np.nanmean(t_hes5[:,jj],axis=1)
    t_hes_n = t_hes_a/np.nanmean(t_hes_a,axis=0)
    plt.plot(t_hes5[:,0]/3600,moving_average(t_hes_a,n=4),'.-',markerfacecolor='none',color='red',alpha=0.2)
    t_ngn2_a = np.nanmean(t_ngn2[:,jj],axis=1)
    t_ngn2_n = t_ngn2_a/np.nanmean(t_ngn2_a,axis=0)
    plt.plot(t_hes5[:,0]/3600,moving_average(t_ngn2_a,n=4),'.-',markerfacecolor='none',color='blue',alpha=0.2)
plt.plot(t/60, P_hes/np.mean(P_hes[t>T*0.8]) , color='red')
plt.plot(t/60, p_NGN/np.mean(p_NGN[t>T*0.8]), color='blue')
plt.ylim([0,3])
plt.xlim([0,25])
plt.legend()

# Find peaks
plt.figure(2)
peaks, _ = find_peaks(p_NGN)
plt.plot(np.diff(t[peaks]/60),'go',label='peaks')
# Find valleys by inverting the data
valleys, _ = find_peaks(-p_NGN)
plt.plot(np.diff(t[valleys])/60,'ro',label='vallyes')
plt.xlabel('Period')
plt.legend()

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

plt.figure(3)
# index = (t>1000) & (t<1200)
# random_ts = np.random.randn(t.shape[0])
# lags_r,cc_r = cross_correlation(random_ts,random_ts,int(t.shape[0]/2))
# plt.plot(lags_r*dt/60,cc_r,'.',markersize=0.1)
# # Fill the area below the curve
# plt.fill_between(lags_r*dt/60, cc_r, color='lightblue', alpha=0.5)
index = (t>1000) & (t<1400)
lags, cc = cross_correlation(p_NGN[index],P_hes[index],int(index.shape[0]/2))
plt.plot(lags*dt/60,cc,'.',markersize=0.2)
plt.grid(True)


