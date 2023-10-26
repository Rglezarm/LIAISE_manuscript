#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:52:26 2023

@author: raquelgonzalezarmas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as final_cases
from matplotlib.dates import DateFormatter
from scipy import optimize
from sklearn.metrics import r2_score
from scipy import stats

#%%
def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

def qsat(T,p):
    return 0.622 * esat(T) / p

#%%
r1    = final_cases.control()

hour_r1 = np.floor(r1.out.t)
minute_r1 = np.floor((r1.out.t-hour_r1)*60)
r1_datetime = [dt.datetime(2021,7,17,int(hour_r1[i]),int(minute_r1[i])) for i in np.arange(len(r1.out.t))]

date_form = DateFormatter("%H:%M")  # Date/Time format
simulations = [r1]
labels = ['Control']
linestyles = ['-']
markers = ['']
time_max = 12
fs = 12
filter_ = (r1.out.t< 15.5)

#%% Load data
# Leaf Chamber data (LiCOR)
LiCOR = pd.read_csv('/Users/raquelgonzalezarmas/OneDrive - Wageningen University &'+\
              ' Research/LIAISE-NLDE/_DATA_EDITED/Ecophys/LC_alfalfa_17_7_2021.csv')
LiCOR['Datetime'] = pd.to_datetime('17/07/2021 '+ \
                        LiCOR['Time'], format='%d/%m/%Y %H:%M:%S')
LiCOR['Time_dt'] = [ dt.datetime.strptime(i, '%H:%M:%S').time() for i in LiCOR['Time'] ]
LiCOR["hour_float_UTC"] = [(float(i[0:2])+float(i[3:5])/60 + float(i[-2:])/3600-2) for i in LiCOR.Time]   # hour in UTC

# In canopy sensors
data_InCanopy = pd.read_csv('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/'+\
                   '1er-paper/observations/InCanopyProfiles/'+\
                       'Canopy_Profile_Temperature_Humidity.csv')
data_InCanopy["DATETIME"] = pd.to_datetime(data_InCanopy.TIMESTAMP, format = '%Y-%m-%d %H:%M:%S')-dt.timedelta(minutes = 2, seconds = 30)
data_InCanopy = data_InCanopy.where(data_InCanopy.DATETIME.dt.day == 17)

data_InCanopy = data_InCanopy.dropna(how = 'any')

heights = [0.105, 0.29, 0.61, 0.99, 1.49]            # Height of sensors inside the canopy in m

# CO2
# EC data
## Load Observations of local energy fluxes
EC_IRGASON = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/OneD'+\
                         'rive-WageningenUniversity&Research/LIAISE-NLDE/_DATA_'+\
                             'EDITED/LaCendrosa/EC/_EddyPro_output/EC150/eddy'+\
                                 'pro_CloudRoots30min_EC150_full_output_2021-'+\
                                     '11-01T154011_adv.csv', header = 1)
    
EC_LiCOR = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/One'+\
                       'Drive-WageningenUniversity&Research/LIAISE-NLDE/_DA'+\
                           'TA_EDITED/LaCendrosa/EC/_EddyPro_output/LiCor750'+\
                               '0/eddypro_CloudRoots30min_LiCor7500_full_ou'+\
                                   'tput_2021-11-01T153930_adv.csv',\
                                       header = 1)

EC_IRGASON, EC_LiCOR = EC_IRGASON[1:], EC_LiCOR[1:]

    
### Choose only the 17/07/2021 which is DOY = 198
EC_IRGASON.DOY = [float(i) for i in EC_IRGASON.DOY] 
EC_IRGASON = EC_IRGASON.where((EC_IRGASON.DOY >= 198) &(EC_IRGASON.DOY < 199))
EC_IRGASON = EC_IRGASON.dropna()
EC_IRGASON["datetime"] = [(dt.datetime.strptime(i, "%Y-%m-%d %H:%M")-  dt.timedelta(minutes = 15))for i in \
                            (EC_IRGASON.date + " " + EC_IRGASON.time)]
    
### Choose only the 17/07/2021 which is DOY = 198
EC_LiCOR.DOY = [float(i) for i in EC_LiCOR.DOY] 
EC_LiCOR = EC_LiCOR.where((EC_LiCOR.DOY >= 198) &(EC_LiCOR.DOY < 199))
EC_LiCOR = EC_LiCOR.dropna()
EC_LiCOR["datetime"] = [(dt.datetime.strptime(i, "%Y-%m-%d %H:%M")-  dt.timedelta(minutes = 15))for i in \
                            (EC_LiCOR.date + " " + EC_LiCOR.time)]
    

xnew0, xold0, xnewspan, xoldspan = 0.01, 3.94, 450.4, 456.1
A = (xnewspan-xnew0)/(xoldspan-xold0)
B = xnew0-xold0*A
CO2_IRGASON = np.array(EC_IRGASON.co2_mixing_ratio, dtype = 'float64')*A+B
CO2_LiCOR = np.array(EC_LiCOR.co2_mixing_ratio, dtype = 'float64')*A+B

CO2_EC = (CO2_IRGASON+CO2_LiCOR)/2
CO2_hour = EC_LiCOR.datetime.dt.hour + EC_LiCOR.datetime.dt.minute/60
#%%

cte = 0.0248

gsw_OBS = LiCOR['gsw[mol.m^-2.s^-1]']
gsw_mean = gsw_OBS.groupby(by = LiCOR.Datetime.dt.hour).mean()*cte
hour_mean = gsw_OBS.groupby(by = LiCOR.Datetime.dt.hour).mean().index - 1.5

# Calculate moving average
## Definition of simple moving average (SMA)
def SMA(x, window):
    n = len(x)
    SMA = np.zeros(n)
    for i in np.arange(n):
        if (i < window) or (i > n - 2 * window):
            SMA[i] = np.nan
        else: 
            SMA[i] = np.sum(x[i-window:i+window+1])/(2*window + 1)
    return SMA
    
## Definition of weighted average
def WMA(x, window, weights):
    n = len(x)
    WMA = np.zeros(n)
    for i in np.arange(n):
        if (i < window) or (i > n - 2 * window):
            WMA[i] = np.nan
        else: 
            WMA[i] = np.sum(weights[i-window:i+window+1]*x[i-window:i+window+1])/(2*window + 1)
    return WMA

def tick_function_gs(gs):
    gs_new = gs / cte
    return ["%.2f" %z for z in gs_new] 


# (3c) Subfigure leaf transpiration 
Lv         = 2.5e6                 # heat of vaporization [J kg-1]
mco2       =  44.;                 # molecular weight CO2 [g mol -1]
mair       =  28.9;                # molecular weight air [g mol -1]
mh2o       = 18.;                  # molecular weight water [g mol-1]
nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide 
from_molm2s_ms = 0.0248

# Calculus of modelled ETleaf
qsatleaf = qsat(r1.out.T0_105m, r1.input.Ps)
qsurf = r1.out.q0_105m
deltaq_leaf = (qsatleaf-qsurf)*1e3             # [g H2O/kg air]
deltaw_leaf = deltaq_leaf[filter_] * mair/mh2o # [mmol water/mol air]
ETleaf = r1.out.glw[filter_]/from_molm2s_ms*deltaw_leaf
ETleaf = r1.out.LEleaf[filter_]/Lv*1e6
# Calculus of observed ETleaf
T_0105m = data_InCanopy.Ta_1_cal + 273.15  # [K]
T_0290m = data_InCanopy.Ta_2_cal + 273.15  # [K]
e_0105m = data_InCanopy.e_1_cal  
e_0290m = data_InCanopy.e_2_cal
RH_0105m = data_InCanopy.RH_1_cal
RH_0290m = data_InCanopy.RH_2_cal

qsat_OBS = qsat(T_0105m, r1.input.Ps)*1e3    # gwater/kgair
qair_OBS = qsat_OBS*RH_0105m               # gwater/kgair 
dw = (qsat_OBS-qair_OBS)*mair/mh2o        # [mmol h2o/mol air]

dw_LiCOR = np.interp(LiCOR.Datetime, data_InCanopy.DATETIME, dw)
TR_OBS = gsw_OBS/nuco2q*dw_LiCOR*mh2o
TR_OBS = gsw_OBS*dw_LiCOR*mh2o
TR_mean = TR_OBS.groupby(by = LiCOR.Datetime.dt.hour).mean()
TR_model = ETleaf*mh2o
TR_model = ETleaf


def tick_function_TR(TR):
    TR_new = TR / mh2o
    return ["%.2f" %z for z in TR_new] 


# (3b) Subfigure leaf assimilation rate
Cair = np.interp(LiCOR.Datetime, CO2_hour, CO2_EC)  #[ppm]
#Cair = 400        # [ppm]
# Calculus of modelled An
An = r1.out.Aln[filter_]/mco2*1e3
# Calculus of observed An
dci = Cair * (1-LiCOR['cica[-]'])   # ppm
An_OBS = gsw_OBS/nuco2q * dci       # micromol/m2/s
An_mean = An_OBS.groupby(by = LiCOR.Datetime.dt.hour).mean()


def tick_function_An(An):
    An_new = An / mco2 *1e3
    return ["%.2f" %z for z in An_new] 



#%% Calculate statistics to compare observations and model results
# (1) Create scatterplots of model results versus observations
## First, we locate the time predictions of the model that are closer to the observation
t_model = r1.out.t[filter_]
glw_model = r1.out.glw[filter_]

filter_OBS = (LiCOR.hour_float_UTC < 15.5)
h_LiCOR = LiCOR.hour_float_UTC[filter_OBS]
gsw_OBS_filtered = gsw_OBS[filter_OBS]
An_OBS_filtered  = An_OBS[filter_OBS]
TR_OBS_filtered  = TR_OBS[filter_OBS]

SMA_gsw_filtered = SMA(x = gsw_OBS, window = 15)[filter_OBS]
SMA_An_filtered = SMA(x = An_OBS, window = 15)[filter_OBS]
SMA_TR_filtered = SMA(x = TR_OBS, window = 15)[filter_OBS]

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

indexes = []
for h in h_LiCOR:
    indexes.append(find_nearest_index(t_model,h))

filter_pred = []
for i in np.arange(len(t_model)):
    if i in indexes:
        filter_pred.append(True)
    else:
        filter_pred.append(False)

def RMSE(x, y):
    RMSE = np.sqrt(np.sum((x-y)**2)/len(x))
    return RMSE

# with scipy
r_gsw, p_gsw = stats.pearsonr(gsw_OBS_filtered*cte, glw_model[filter_pred])
RMSE_gsw = RMSE(gsw_OBS_filtered*cte, glw_model[filter_pred])

# with scipy
r_An, p_An = stats.pearsonr(An_OBS_filtered, An[filter_pred])
RMSE_An = RMSE(An_OBS_filtered*mco2*1e-3, An[filter_pred]*mco2*1e-3)

# with scipy
r_TR, p_TR = stats.pearsonr(TR_OBS_filtered, TR_model[filter_pred])
RMSE_TR = RMSE(TR_OBS_filtered, TR_model[filter_pred])

#%% Plot diurnal evolution of model results and observations
fig, axs = plt.subplots(1, 3, dpi = 350, figsize = (14,5))
ax1, ax3, ax2 = axs[0], axs[1], axs[2]

# (3a) Subfigure stomatal conductance
ax1.set_title(r'(4a) $\mathrm{g_{s}}$')
ax1.grid()
cte = 0.0248

ax1.plot(r1.out.t[filter_], r1.out.glw[filter_], color = 'k', label = r'A-g$_s$, CONTROL', zorder = 15)
ax1.scatter(LiCOR.hour_float_UTC, gsw_OBS*cte, alpha = 0.6, label = 'OBS', zorder = 10)
#ax1.plot(hour_mean, gsw_mean,  marker = 'o', markersize = 2, color ='C1', label = 'mean OBS', zorder = 15)
ax1.plot(LiCOR.hour_float_UTC, SMA(x = gsw_OBS, window = 15)*cte, color ='blue', label = 'moving averaged OBS', zorder = 15)
#ax1.scatter(LiCOR.hour_float_UTC, WMA(x = gsw_OBS, window = 15)*cte,  marker = 'o', color ='grey', label = 'weighted moving average OBS', zorder = 15)
ax1.set_xlabel('Time [UTC]',fontsize = fs)
ax1.set_ylabel(r'm s$^{-1}$', fontsize = fs)
ax1.set_xlim(4,20)
ax1.axvline(12+2/60+27/3600, color = 'orange', linewidth = 1., zorder = 2)
ax1.legend(bbox_to_anchor=(0.8,-0.2))
#ax1.annotate(r'$r^2$ = %.3f [-]'%r_gsw**2 + '\n' + 'p-value = %.1e [-]'%p_gsw + '\n'\
 #           + 'RMSE = %.3f [m s$^{-1}$]'%RMSE_gsw, (6.5, 0.002))
ax1.annotate(r'$r^2$ = %.3f***'%r_gsw**2 + '\n'\
            + 'RMSE = %.3f m s$^{-1}$'%RMSE_gsw, (6.5, 0.002))
x_axis_ticks = np.arange(4,22,2)
ax1.set_xticks(x_axis_ticks)
## add second axis
ax1b = ax1.twinx()

new_tick_locations_gs = np.array([0.005, 0.010, 0.015, 0.02,0.025, 0.03]) #p.array([0.,2.,4.,6.,8.,10.,12.])

ax1b.set_ylim(ax1.get_ylim())
ax1b.set_yticks(new_tick_locations_gs)
ax1b.set_yticklabels(tick_function_gs(new_tick_locations_gs))
ax1b.set_ylabel(r'mol$_{air}~m_{leaf}^{-2}~s^{-1}$', fontsize = fs)

ax1.axvline(LiCOR.hour_float_UTC[np.where(SMA_gsw_filtered == np.nanmax(SMA_gsw_filtered))[0][0]], color = 'red', linestyle = '--')
ax1.axvline(r1.out.t[filter_][np.where(r1.out.glc[filter_] == np.nanmax(r1.out.glc[filter_]))[0][0]], color = 'red', linestyle = '-')
# (3c) Subfigure leaf transpiration 
Lv         = 2.5e6                 # heat of vaporization [J kg-1]
mco2       =  44.;                 # molecular weight CO2 [g mol -1]
mair       =  28.9;                # molecular weight air [g mol -1]
mh2o       = 18.;                  # molecular weight water [g mol-1]
nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide 
from_molm2s_ms = 0.0248

ax2.set_title(r'(4c) $\mathrm{TR_{leaf}}$')
ax2.plot(r1.out.t[filter_], TR_model, c = 'k', label = None, zorder = 15)
ax2.scatter(LiCOR.hour_float_UTC, TR_OBS, alpha = 0.6, label = 'post-processed OBS', marker = 'x', zorder = 10)
#ax2.plot(hour_mean, TR_mean,  marker = 'o', markersize = 2, color ='C1', label = 'mean OBS', zorder = 15)
ax2.plot(LiCOR.hour_float_UTC, SMA(x = TR_OBS, window = 15),  color ='blue',alpha = 0.9, label = 'moving averaged post-processed OBS', zorder = 15)
ax2.axvline(12+2/60+27/3600, color = 'orange', linewidth = 1., zorder = 2)
ax2.set_xlabel('Time [UTC]',fontsize = fs)
ax2.set_ylabel(r'mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$',fontsize = fs)
ax2.set_xlim(4,20)
ax2.set_ylim(-30,350)
ax2.grid()
ax2.set_xticks(x_axis_ticks)
ax2.axvline(LiCOR.hour_float_UTC[np.where(SMA_TR_filtered == np.nanmax(SMA_TR_filtered))[0][0]], color = 'red', linestyle = '--')
ax2.axvline(r1.out.t[filter_][np.where(r1.out.LEleaf[filter_] == np.nanmax(r1.out.LEleaf[filter_]))[0][0]], color = 'red', linestyle = '-')
## add second axis
ax2b = ax2.twinx()

new_tick_locations_TR = np.array([0,50,100,150,200,250,300]) 

ax2b.set_ylim(ax2.get_ylim())
ax2b.set_yticks(new_tick_locations_TR)
ax2b.set_yticklabels(tick_function_TR(new_tick_locations_TR))
ax2b.set_ylabel(r'mmol$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$', fontsize = fs)

ax2.annotate(r'$r^2$ = %.3f***'%r_TR**2 + '\n'\
            + 'RMSE = %d mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$'%RMSE_TR, (6.5, -18))
# (3b) Subfigure leaf assimilation rate

ax3.plot(r1.out.t[filter_], An*mco2*1e-3, color = 'k', label = None, zorder = 15)
ax3.scatter(LiCOR.hour_float_UTC, An_OBS*mco2*1e-3, alpha = 0.6, label = 'post-processed OBS', marker = 'x', zorder = 10)
#ax3.plot(hour_mean, An_mean*mco2*1e-3,  marker = 'o', markersize = 2, color ='C1', label = 'mean OBS', zorder = 15)
ax3.plot(LiCOR.hour_float_UTC, SMA(x = An_OBS, window = 15)*mco2*1e-3, color ='blue', label = 'moving averaged post-processed OBS', zorder = 15, alpha = 0.9)
ax3.axvline(12+2/60+27/3600, color = 'orange', linewidth = 1., zorder = 2)
ax3.set_title(r'(4b) $\mathrm{A_n}$')
ax3.set_xlabel('Time [UTC]',fontsize = fs)
ax3.set_ylabel(r'mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$ ',fontsize = fs)
ax3.set_xlim(4,20)
ax3.annotate(r'$r^2$ = %.3f***'%r_An**2 + '\n'\
            + 'RMSE = %.1f mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$'%RMSE_An, (6.2, -0.35))
ax3.grid()
ax3.set_ylim(-0.5,2.8)
ax3.legend(bbox_to_anchor=(1.1,-0.2))
ax3.set_xticks(x_axis_ticks)
ax3.axvline(LiCOR.hour_float_UTC[np.where(SMA_An_filtered == np.nanmax(SMA_An_filtered))[0][0]], color = 'red', linestyle = '--')
ax3.axvline(r1.out.t[filter_][np.where(r1.out.Aln[filter_] == np.nanmax(r1.out.Aln[filter_]))[0][0]], color = 'red', linestyle = '-')
## add second axis
ax3b = ax3.twinx()

new_tick_locations_An = np.array([0,0.5,1.,1.5,2.,2.5]) 

ax3b.set_ylim(ax3.get_ylim())
ax3b.set_yticks(new_tick_locations_An)
ax3b.set_yticklabels(tick_function_An(new_tick_locations_An))
ax3b.set_ylabel(r'$\mu$mol$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$', fontsize = fs)


plt.tight_layout()
plt.show()

#%% EXTRA (not present in the paper) Make scatter plots with statistics
fig, axs = plt.subplots(1, 3, dpi = 350, figsize = (12,4))
ax1, ax2, ax3 = axs[0], axs[1], axs[2]

x1 = np.array([0,0.01,0.02,0.03])
ax1.scatter(gsw_OBS_filtered*cte, glw_model[filter_pred], label = None, alpha = 0.7)
ax1.plot(x1, x1, label = '1:1 line', c = 'k')
ax1.set_xlabel(r'Observed g$_{sw}$ [m s$^{-1}$]')
ax1.set_ylabel(r'Predicted g$_{sw}$ [m s$^{-1}$]')

ax1.annotate(r'$r^2$ = %.3f [-]'%r_gsw**2 + '\n' + 'p-value = %.1e [-]'%p_gsw + '\n'\
            + 'RMSE = %.3f [m s$^{-1}$]'%RMSE_gsw, (0.00, 0.023))
    
# Fit a linear regression
m_gsw,b_gsw = np.polyfit(gsw_OBS_filtered*cte, glw_model[filter_pred], 1)
ax1.plot(x1, m_gsw*x1 + b_gsw, linestyle = '--', color = 'k')
ax1.annotate(r'slope = %.2f'%m_gsw**2 + '\n' + 'intercept = %.3f m s$^{-1}$'%b_gsw, (0.002, -0.001))
ax1.grid()
ax1.legend(loc = 'lower right')

ax2.scatter(An_OBS_filtered*mco2*1e-3, An[filter_pred]*mco2*1e-3, label = None, alpha = 0.7)
x2 = np.array([0, 1, 2.5 ])
ax2.plot(x2,x2, label = '1:1 line', c = 'k')
ax2.set_xlabel(r'Observed A$_{n}$ [mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$]')
ax2.set_ylabel(r'Predicted A$_{n}$ [mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$]')

ax2.annotate(r'$r^2$ = %.3f '%r_An**2 + '\n' + 'p-value = %.1e'%p_An + '\n'\
            + 'RMSE = %.1f mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$'%RMSE_An, (0.00, 2.1))
# Fit a linear regression
m_An,b_An = np.polyfit(An_OBS_filtered*mco2*1e-3, An[filter_pred]*mco2*1e-3, 1)
ax2.plot(x2, m_An*x2 + b_An, linestyle = '--', color = 'k')
ax2.annotate(r'slope = %.2f'%m_An**2 + '\n' + 'intercept = %.3f mg$_{CO_2}$ m$_{leaf}^{-2}$  s$^{-1}$'%b_An, (0.3, -0.1))
ax2.grid()
ax2.legend(loc = 'lower right')
ax2.set_ylim(-0.5,2.7)

## Leaf transpiration
x3 = np.array([0, 300 ])
ax3.scatter(TR_OBS_filtered, TR_model[filter_pred], label = None, alpha = 0.7)
ax3.plot(x3,x3, label = '1:1 line', c = 'k')
ax3.set_xlabel(r'Observed TR$_{leaf}$ [mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$]')
ax3.set_ylabel(r'Predicted TR$_{leaf}$ [mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}]$')

ax3.annotate(r'$r^2$ = %.3f'%r_TR**2 + '\n' + 'p-value = %.1e'%p_TR + '\n'\
            + 'RMSE = %d mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$'%RMSE_TR, (0.00, 300))

# Fit a linear regression
m_TR,b_TR = np.polyfit(TR_OBS_filtered, TR_model[filter_pred], 1)
ax3.plot(x3, m_TR*x3 + b_TR, linestyle = '--', color = 'k')
ax3.annotate(r'slope = %.2f'%m_TR**2 + '\n' + 'intercept = %.3f'%b_TR + '\n' + 'mg$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$', (120, 30))
    
ax3.grid()
ax3.legend(loc = 'lower right')

fig.tight_layout()

"""
### Create a histogram with the residuals of the fitted least SS linear regression
residuals = glw_model[filter_pred] - (alpha[0]*gsw_OBS_filtered*cte+alpha[1])

fig = plt.figure(dpi = 350)
ax = fig.gca()

ax.hist(residuals, bins = 30)
ax.set_ylabel('Frequency')
ax.set_xlabel(r'Residuals [m s$^{-1}$]')
ax.set_xticks(np.array([-0.01, -0.005, 0, 0.005, 0.01]))
"""

