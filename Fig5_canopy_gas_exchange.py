#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:55:23 2023

@author: raquelgonzalezarmas
"""

"""
    ===> Figure 5 of the LIAISE paper
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/1er-paper/models/')
from matplotlib.dates import DateFormatter
import sensitivity_control_cases as final_cases
import matplotlib.gridspec as gridspec
from scipy import stats

#%% OBSERVATIONS
## Load Observations of Local energy fluxes and radiation
fluxes = pd.read_csv('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/'+\
                     '1er-paper/observations/corrected_fluxes_17_07_2'+\
                         '021_LaCendrosa.csv')
hour_fluxes = np.floor(fluxes.hour_dec)
minute_fluxes = np.floor((fluxes.hour_dec-hour_fluxes)*60)
fluxes_datetime = [dt.datetime(2021,7,17,int(hour_fluxes[i]),int(minute_fluxes[i])) for i in np.arange(len(hour_fluxes))]

## Load Observations of local radiation
radiation_local = pd.read_csv('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/'+\
                              '1er-paper/observations/2021-07-11_to_2021-08-01_'+\
                                  'LaCendrosa_RadSoil_Nov26.csv')
radiation_local.DOY = [float(i) for i in radiation_local.DOY] 
indexes = radiation_local.index[radiation_local['DOY'] != 198].tolist()
radiation_local = radiation_local.drop(index = indexes)
radiation_local["hour_float"] = [(float(i[:2]) + float(i[-2:])/60) for \
                                 i in radiation_local["HH:MM"]]
## Load Observations of soil respiration
soil = pd.read_excel('/Users/raquelgonzalezarmas/Library/CloudStorage/'+\
                          'OneDrive-WageningenUniversity&Research/PhD_Wagen'+\
                              'ingen/1er-paper/observations/Soil_respiration/'+\
                                  'La_Cendrosa_Soil_respiration_LIAISE_2021.'+\
                                      'xlsx')

soil_intervals = soil['Time_intervals']
soil_resp = soil['Q (mumol m-2 s1)']   # micromol CO2 m-2 s-1
soil_hour_dec = np.array([(soil.Time[i].hour-2 + soil.Time[i].minute/60 + soil.Time[i].second/3600) for i in np.arange(len(soil_resp))])

#%% Average soil respiration measurements, calculate its mean hour in UTC and calculate standard deviation

rho        = 1.2                   # density of air [kg m-3]
mair       =  28.9;                # molecular weight air [g mol -1]
mco2       =  44.                  # molecular weight CO2 [g mol -1]
mh2o       =  18.                  # molecular weight H2O [g mol -1]
Lv         = 2.5e6                 # heat of vaporization [J kg-1]

fac = mair / (rho*mco2)


len_mean_Resp = len(np.unique(soil_intervals))
mean_Resp = np.zeros(len_mean_Resp)
std_Resp = np.zeros(len_mean_Resp)
hour_Resp = np.zeros(len_mean_Resp)
SE_Resp = np.zeros(len_mean_Resp)
for i in np.arange(len_mean_Resp):
    n =  np.count_nonzero(soil_intervals == np.unique(soil_intervals)[i])
    mean_Resp[i] = np.mean(soil_resp[soil_intervals == np.unique(soil_intervals)[i]])*mco2*1e-3    
    std_Resp[i] = np.std(soil_resp[soil_intervals == np.unique(soil_intervals)[i]])*mco2*1e-3    
    hour_Resp[i] = np.mean(soil_hour_dec[soil_intervals == np.unique(soil_intervals)[i]])
    SE_Resp[i] = std_Resp[i]/np.sqrt(2*n-2)

Resp = mean_Resp
#%%
r1     = final_cases.control()

hour_r1 = np.floor(r1.out.t)
minute_r1 = np.floor((r1.out.t-hour_r1)*60)
r1_datetime = [dt.datetime(2021,7,17,int(hour_r1[i]),int(minute_r1[i])) for i in np.arange(len(r1.out.t))]

#%% Figure settings
date_form = DateFormatter("%H:%M")  # Date/Time format
simulations = [r1]
labels = ['Control']
linestyles = ['-']
markers = ['']
time_max = 12
fs = 14

#%% Calculate statistics of the data

# changing units and filtering data



x = r1.out.t
ET_CLASS = r1.out.LE/Lv*1e6
NEE_CLASS = r1.out.wCO2  # [mg CO2 m-2 s-1]
Resp_CLASS = r1.out.wCO2R  # [mg CO2 m-2 s-1]
Anc_CLASS = r1.out.wCO2A  # [mg CO2 m-2 s-1]
    
filter_ = (x>15.5)
ET_CLASS[filter_] = np.nan
NEE_CLASS[filter_] = np.nan
Resp_CLASS[filter_] = np.nan
Anc_CLASS[filter_] = np.nan

## Energy balance gap
Rn = np.array(radiation_local.Rn)
H_ = np.array(fluxes.H)
G_ = np.array(fluxes.G)
LE_ = np.array(fluxes.LE)
EBG = Rn - H_ - G_ - LE_

ET = LE_/Lv*1e6
NEE = fluxes.co2_flux*mco2/1e3

# Calculate statistics on the data

def RMSE(x, y):
    RMSE = np.sqrt(np.sum((x-y)**2)/len(x))
    return RMSE

# Filter observations
filter_fluxes = (fluxes.hour_dec>= 5)&(fluxes.hour_dec <= 15.5)
hour_fluxes = fluxes.hour_dec[filter_fluxes]
NEE_filtered = NEE[filter_fluxes]
ET_filtered = ET[filter_fluxes]
filter_Resp = hour_Resp <= 15.5
Resp_filtered = Resp[filter_Resp]
hour_Resp2 = hour_Resp[filter_Resp]

# Calculate model predicion for the same time when observations were performed
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

indexes_fluxes = []
indexes_Resp   = []

for h in hour_fluxes:
    indexes_fluxes.append(find_nearest_index(r1.out.t,h))

for h in hour_Resp2:
    indexes_Resp.append(find_nearest_index(r1.out.t,h))
    
filter_pred_fluxes = []
for i in np.arange(len(r1.out.t)):
    if i in indexes_fluxes:
        filter_pred_fluxes.append(True)
    else:
        filter_pred_fluxes.append(False)

filter_pred_Resp = []
for i in np.arange(len(r1.out.t)):
    if i in indexes_Resp:
        filter_pred_Resp.append(True)
    else:
        filter_pred_Resp.append(False)

NEE_CLASS_filtered = NEE_CLASS[filter_pred_fluxes]
Anc_CLASS_filtered = Anc_CLASS[filter_pred_fluxes]
ET_CLASS_filtered = ET_CLASS[filter_pred_fluxes]
Resp_CLASS_filtered = Resp_CLASS[filter_pred_Resp]

# with scipy
r_NEE, p_NEE = stats.pearsonr(NEE_filtered, NEE_CLASS_filtered)
RMSE_NEE = RMSE(NEE_filtered, NEE_CLASS_filtered)

# with scipy
r_Resp, p_Resp = stats.pearsonr(Resp_filtered, Resp_CLASS_filtered)
RMSE_Resp = RMSE(Resp_filtered, Resp_CLASS_filtered)

# with scipy
r_ET, p_ET = stats.pearsonr(ET_filtered, ET_CLASS_filtered)
RMSE_ET = RMSE(ET_filtered, ET_CLASS_filtered)

#%%
fig, axs = plt.subplots(1, 2, dpi = 350, figsize = (14,5))
ax1, ax2 = axs[0], axs[1]
    
ax2.plot(r1.out.t, ET_CLASS,  color = 'black', \
                  ls  = linestyles[0],  label = 'ET, CONTROL', zorder = 10)
ax1.plot(r1.out.t, NEE_CLASS,  color = 'black', \
                  ls  = linestyles[0],  label = 'NEE, CONTROL', zorder = 10)
ax1.plot(r1.out.t, Resp_CLASS,  color = 'black', \
            ls  = '--',  label = 'Resp., CONTROL', zorder = 10)
ax1.plot(r1.out.t, Anc_CLASS,  color = 'black', \
             ls  = ':',  label = '-A$_{nc}$, CONTROL', zorder = 10)
## Observations
ax2.scatter(fluxes.hour_dec, fluxes.LE/Lv*1e6, label = 'ET, OBS', zorder = 5)

###(3) Sensible Heat Flux
ax2.fill_between(fluxes.hour_dec, ET,ET+EBG/Lv*1e6, color = '#1f77b4',  alpha = 0.3, label = 'ET + Imb, OBS',zorder = 5)
ax1.scatter(fluxes.hour_dec, NEE, zorder = 5, label = 'NEE, OBS')
ax1.scatter(hour_Resp, Resp, color = 'brown', label = 'Resp. OBS')

ax1.legend(loc = 'center', bbox_to_anchor=(0.5,-0.3), ncol = 2)
ax1.errorbar(hour_Resp, mean_Resp, yerr=SE_Resp, fmt="o", color="brown")
## Settings
# add second y axis for NEE, GPP and Resp figure (4a)
ax1b = ax1.twinx()

def tick_function_wCO2(wCO2_mg):
    wCO2_mu = wCO2_mg/mco2*1e3
    return ["%.1f" %z for z in wCO2_mu] 

new_tick_locations3 = np.array([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4])

ax1b.set_ylim(ax1.get_ylim())
ax1b.set_yticks(new_tick_locations3)
ax1b.set_yticklabels(tick_function_wCO2(new_tick_locations3))
ax1b.set_ylabel(r'$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$', fontsize = fs)



#ax1.xaxis.set_major_formatter(date_form)

ax2.set_title('(5b) ET', fontsize = fs)

ax2.set_xlabel('Time [UTC]', fontsize = fs)
ax2.set_ylabel(r'mg H$_2$O m$^{-2}$ s$^{-1}$', fontsize = fs)
ax2.set_xlim(4,20)
ax2.grid()
ax2.legend(loc = 'lower center', bbox_to_anchor=(0.5,-0.45))
ax1.set_title('(5a) NEE, -A$_{nc}$ and Resp.', fontsize = fs)
ax1.set_xlim(4,20)
ax1.set_xlabel('Time [UTC]', fontsize = fs)
ax1.set_ylabel(r'mg CO$_2$ m$^{-2}$ s$^{-1}$', fontsize = fs)
ax1.annotate(r'$r^2_{NEE}$ = %.3f***'%r_NEE**2 + '\n'\
            + 'RMSE$_{NEE}$ = %.3f mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$'%RMSE_NEE, (9, -0.2))
ax2.annotate(r'$r^2$ = %.3f***'%r_ET**2 + '\n'\
            + 'RMSE = %d mg$_{H_2O}$ m$_{surf}^{-2}$  s$^{-1}$'%RMSE_ET, (10, 20))
ax1.grid()

ax1.axvline(12+2/60+27/3600, color = 'orange', linewidth = 1., zorder = 2)
ax1.axvline(r1.out.t[np.where(NEE_CLASS == np.nanmin(NEE_CLASS))[0][0]], color = 'red', linestyle = '-')
ax1.axvline(fluxes.hour_dec[np.where(NEE == np.nanmin(NEE))[0][0]], color = 'red', linestyle = '--')
ax2.axvline(12+2/60+27/3600, color = 'orange', linewidth = 1., zorder = 2)
ax2.axvline(r1.out.t[np.where(ET_CLASS == np.nanmax(ET_CLASS))[0][0]], color = 'red', linestyle = '-')
ax2.axvline(fluxes.hour_dec[np.where(ET == np.nanmax(ET))[0][0]], color = 'red', linestyle = '--')

# add second y axis for ET/LE figure (4b)
ax2b = ax2.twinx()

def tick_function(ET):
    ET_mumol = ET /mh2o 
    return ["%d" %z for z in ET_mumol] 
new_tick_locations1 = np.array([0.,50,100,150,200])

ax2b.set_ylim(ax2.get_ylim())
ax2b.set_yticks(new_tick_locations1)
ax2b.set_yticklabels(tick_function(new_tick_locations1))
ax2b.set_ylabel(r'mmol H$_2$O m$^{-2}$ s$^{-1}$', fontsize = fs)

# add a third y axis for ET/LE figure (4b)
ax2c = ax2.twinx()
ax2c.spines['right'].set_position(("axes", 1.2))

def tick_function2(ET):
    LE = ET * Lv/1e6
    return ["%d" %z for z in LE] 
new_tick_locations2 = np.array([0.,50,100,150,200])

ax2c.set_ylim(ax2.get_ylim())
ax2c.set_yticks(new_tick_locations2)
ax2c.set_yticklabels(tick_function2(new_tick_locations2))
ax2c.set_ylabel(r'W m$^{-2}$', fontsize = fs)

fig.tight_layout()
#order = [3,0,1,2]
#handles, labels_ = ax1.get_legend_handles_labels()
#first_legend = plt.legend([handles[idx] for idx in order],[labels_[idx] for idx in order], loc='lower center', bbox_to_anchor=(1.2,-0.28)), ncol = 4)


#%% EXTRA (not shown in the manuscript) Scatter-plots with statistics
fig, axs = plt.subplots(1, 2, dpi = 350, figsize = (8,4))
ax1, ax3 = axs[0], axs[1]

x1 = np.array([-1, 0.4])
ax1.scatter(NEE_filtered, NEE_CLASS_filtered, label = None, alpha = 0.7)
ax1.plot(x1,x1, label = '1:1 line', c = 'k')
ax1.set_xlabel(r'Observed NEE [mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$]')
ax1.set_ylabel(r'Predicted NEE [mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$]')

ax1.annotate(r'$r^2$ = %.3f'%r_NEE**2 + '\n' + 'p-value = %.1e'%p_NEE + '\n'\
            + 'RMSE = %.3f mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$'%RMSE_NEE, (-1, 0.2))
# Fit a linear regression
m_NEE,b_NEE = np.polyfit(NEE_filtered, NEE_CLASS_filtered, 1)
ax1.plot(x1, m_NEE*x1 + b_NEE, linestyle = '--', color = 'k')
ax1.annotate(r'slope = %.2f'%m_NEE**2 + '\n' + 'intercept = %.3f'%b_NEE + '\n' + 'mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$', (-0.4, -0.8))

ax1.grid()
ax1.legend(loc = 'lower right')

"""
-- No statistics performed for soil respiration due to scarcity of data points
x2 = np.array([0,0.4])
ax2.scatter(Resp_filtered, Resp_CLASS_filtered, label = None, alpha = 0.7)
ax2.plot(x2,x2, label = '1:1 line', c = 'k')
ax2.set_xlabel(r'Observed Resp. [mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$]')
ax2.set_ylabel(r'Predicted Resp [mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$]')

ax2.annotate(r'$r^2$ = %.3f [-]'%r_Resp**2 + '\n' + 'p-value = %.1e [-]'%p_Resp + '\n'\
            + 'RMSE = %.3f [mg$_{CO_2}$ m$_{surf}^{-2}$  s$^{-1}$]'%RMSE_Resp, (0.12, 0.05))
ax2.grid()
ax2.legend(loc = 'lower right')
"""

x3 = np.array([0,200])
ax3.scatter(ET_filtered, ET_CLASS_filtered, label = None, alpha = 0.7)
ax3.plot(x3,x3, label = '1:1 line', c = 'k')
ax3.set_xlabel(r'Observed ET [mg$_{H_2O}$ m$_{surf}^{-2}$  s$^{-1}$]')
ax3.set_ylabel(r'Predicted ET [mg$_{H_2O}$ m$_{surf}^{-2}$  s$^{-1}$]')

ax3.annotate(r'$r^2$ = %.3f'%r_ET**2 + '\n' + 'p-value = %.1e'%p_ET + '\n'\
            + 'RMSE = %d mg$_{H_2O}$ m$_{surf}^{-2}$  s$^{-1}$'%RMSE_ET, (0, 230))
# Fit a linear regression
m_ET,b_ET = np.polyfit(ET_filtered, ET_CLASS_filtered, 1)
ax3.plot(x3, m_ET*x3 + b_ET, linestyle = '--', color = 'k')
ax3.annotate(r'slope = %.2f'%m_ET**2 + '\n' + 'intercept = %d'%b_ET + '\n' + 'mg$_{H_2O}$ m$_{surf}^{-2}$  s$^{-1}$', (100, 30))
    
ax3.grid()
ax3.legend(loc = 'lower right')

fig.tight_layout()


#%% EXTRA (not shown in the paper) Quantify average energy budget non-closure from 7 to 17 UTC

# Time series of energy budget non-closure
fig = plt.figure(dpi=350)
ax  = fig.gca()

ax.plot(fluxes.hour_dec, EBG)
ax.set_xlabel('Time [UTC]')
ax.set_ylabel(r'Energy budget non-closure [W m$^{-2}$]')

# Time series of percentual energy budget non-closure witn respect to available energy
fig = plt.figure(dpi=350)
ax  = fig.gca()

ax.plot(fluxes.hour_dec, EBG/(Rn+G_)*100)
ax.axvline(7, c = 'k')
ax.axvline(17, c = 'k')
ax.set_xlabel('Time [UTC]')
ax.set_ylabel(r'Energy budget non-closure [%]')

mean_EBG = np.nanmean((EBG/(Rn+G_)*100)[(fluxes.hour_dec>7)&(fluxes.hour_dec<17)])