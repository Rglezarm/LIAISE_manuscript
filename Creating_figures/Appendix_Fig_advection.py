#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:35:19 2022

@author: raquelgonzalezarmas
"""


"""
    ===> Script that compares observed advection with advectioned modelled in CLASS.
        (1) Observed advection was derived by Mary Rose for the 17/07/2021
        (2) Model advection was calculated in two different settings:
            (a) Only simulating the noon advection
            (b) Simulating the three observed periods of advection: (1) early morning, (2) midday and (3) late evening sea breeze
    for LIAISE 17/07/2022 experiment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import DateFormatter
import sys
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/Code/Numerical-Methods')
import step_functions as step

# Load advection data and select only the day of interest for us
T_adv = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/'+\
                    '1er-paper/observations/LC_adv_T.csv', usecols = \
                        ['mean_adv','TIMESTAMP'])
T_adv["DATETIME"] = pd.to_datetime(T_adv.TIMESTAMP)
T_adv = T_adv[T_adv.DATETIME.dt.day == 17]
T_adv["hour_dec"] = T_adv.DATETIME.dt.hour + T_adv.DATETIME.dt.minute/60
q_adv = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/'+\
                    '1er-paper/observations/LC_adv_q.csv', usecols = \
                        ['mean_adv','TIMESTAMP'])
q_adv["DATETIME"] = pd.to_datetime(q_adv.TIMESTAMP)
q_adv = q_adv[q_adv.DATETIME.dt.day == 17]
q_adv["hour_dec"] = q_adv.DATETIME.dt.hour + q_adv.DATETIME.dt.minute/60
#%% Figure of temperature advection
fig = plt.figure(dpi = 400)
ax = fig.gca()

ax.scatter(T_adv.hour_dec, T_adv.mean_adv, label = 'OBS estimation', c = 'k', s = 4)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel(r'Adv$_{\theta}$ (K $\mathrm{h^{-1}}$)')
ax.grid()

### Model 1
time = np.arange(5,21,0.1)                                   # Hour in UTC
T_adv_cte = 5/(6)     # K / s
T_adv_model = step.Flan(time, T_adv_cte, 0.5, 9,17)
q_adv_cte = -0.0003 * 1/(15)*1000                           # kg water/kg air /s
q_adv_model = step.Flan(time, q_adv_cte, 0.5, 11,18)

### Model 2
#### Heat
T_adv_cte_1 = 0.8
T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
T_adv_cte_2 = 1
T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
T_adv_model_1_2 = T_adv_model_1 + T_adv_model_2 
#### Moisture
q_adv_cte_1 = -0.15
q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
q_adv_cte_2 = -0.0003 * 1/(15)*1000   
q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
q_adv_model_1_2 = q_adv_model_1 + q_adv_model_2 

### Model 3
#### Heat
T_adv_cte_1 = 0.8
T_adv_model_1 = step.Flan(time, T_adv_cte, 1, 5,6.5)
T_adv_cte_2 = 1
T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
T_adv_cte_3 = -3
T_adv_model_3 = step.Flan_skewed(x = time, amplitude =T_adv_cte_3, k1 = 6, k2 = 1, a= 18.45, b= 19.4)
T_adv_model_1_2_3 = T_adv_model_1 + T_adv_model_2 + T_adv_model_3
#### Moisture
q_adv_cte_1 = -0.15
q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
q_adv_cte_2 = -0.0003 * 1/(15)*1000   
q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
q_adv_cte_3 = 0.1 
q_adv_model_3 = step.Flan_skewed(x = time, amplitude = q_adv_cte_3, k1=3.9, k2 = 2, a= 18.7,b=19.5)
q_adv_model_1_2_3 = q_adv_model_1 + q_adv_model_2 + q_adv_model_3

### Model 2 + extra 
#### Heat
T_adv_cte_1 = 0.8
T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
T_adv_cte_2 = 2
T_adv_model_2_extra = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
T_adv_model_1_2_extra = T_adv_model_1 + T_adv_model_2_extra 


#ax.plot(time, T_adv_model, label = 'Model 1', c = 'k')
#ax.plot(time, T_adv_model_1_2, label = 'Control advection', c = 'k', linestyle = '-')
#ax.plot(time, T_adv_model_1_2_3, label = 'Model 3', c = 'k', linestyle = ':')
ax.plot(time, T_adv_model_1_2, label = 'BC model', c = 'k', linestyle = '--')
ax.set_xlim(5,21)
ax.set_ylim(-2.5,2.5)
ax.legend()
#%% Figure of specific humidity advection
fig = plt.figure(dpi = 400)
ax = fig.gca()

ax.scatter(q_adv.hour_dec, q_adv.mean_adv, label = 'OBS estimation', c = 'k', s = 4)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel(r'Adv$_q$ ($\mathrm{g_{water}~kg_{air}^{-1}~ h^{-1}}$)')
ax.grid()
#ax.plot(time, q_adv_model, label = 'Model 1', c = 'k')
ax.plot(time, q_adv_model_1_2, label = 'BC model' , c = 'k', linestyle = '--')
#ax.plot(time, q_adv_model_1_2_3, label = 'Model 3' , c = 'k', linestyle = ':')
ax.set_xlim(5,21)
ax.legend()




