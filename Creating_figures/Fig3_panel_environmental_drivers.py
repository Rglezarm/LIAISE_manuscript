#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:37:50 2023

@author: raquelgonzalezarmas
"""


"""   ===> Script that creates Figure 2 of LIAISE paper
        Option 1: PAR figure in the row above, theta and VPD in the row below
"""

# Import packages
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter
from matplotlib import cm
import sys
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as cases
#%% Open observations

# EC data
## Load Observations of local energy fluxes
EC_data = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/1er-paper'+\
                           '/observations/eddypro_LACENDROSA_Alfalfa_30min'+\
                               '_full_output_2022-04-25T143528_adv.csv', header = 1)
EC_data = EC_data[1:]

    
### Choose only the 17/07/2021 which is DOY = 198
EC_data.DOY = [float(i) for i in EC_data.DOY] 
EC_data = EC_data.where((EC_data.DOY >= 198) &(EC_data.DOY < 199))
EC_data = EC_data.dropna()
EC_data["datetime"] = [(dt.datetime.strptime(i, "%Y-%m-%d %H:%M")-  dt.timedelta(minutes = 15))for i in \
                            (EC_data.date + " " + EC_data.time)]
#EC_data["hour_float"] = [i.hour + i.minute/60 -15/60 for i in EC_data.datetime]
## Open 50 m tower observations, La Cendrosa
data_tower = xr.open_dataset('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/1er-paper/observations/tower-50m/LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_2021-07-17_V1.00.nc')

data_tower['time_'] = [(dt.datetime(data_tower.time[i].dt.year.item(),data_tower.time[i].dt.month.item(),\
                                   data_tower.time[i].dt.day.item(),data_tower.time[i].dt.hour.item(),\
                                       data_tower.time[i].dt.minute.item(),data_tower.time[i].dt.second.item())\
                                   -dt.timedelta(minutes = 15)) for i in np.arange(len(data_tower.time))]

## Open in canopy observations (temperature and humimdity), La Cendrosa
data_InCanopy = pd.read_csv('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/'+\
                   '1er-paper/observations/InCanopyProfiles/'+\
                       'Canopy_Profile_Temperature_Humidity.csv')
data_InCanopy["DATETIME"] = pd.to_datetime(data_InCanopy.TIMESTAMP, format = '%Y-%m-%d %H:%M:%S')-dt.timedelta(minutes = 2, seconds = 30)

data_InCanopy = data_InCanopy.where(data_InCanopy.DATETIME.dt.day == 17)
heights = [0.105, 0.29, 0.61, 0.99, 1.49]            # Height of sensors inside the canopy in m

T = [data_InCanopy.Ta_1_cal, data_InCanopy.Ta_2_cal, data_InCanopy.Ta_3_cal, data_InCanopy.Ta_4_cal, data_InCanopy.Ta_5_cal]
e = [data_InCanopy.e_1_cal, data_InCanopy.e_2_cal, data_InCanopy.e_3_cal, data_InCanopy.e_4_cal, data_InCanopy.e_5_cal]
RH = [data_InCanopy.RH_1_cal, data_InCanopy.RH_2_cal, data_InCanopy.RH_3_cal, data_InCanopy.RH_4_cal, data_InCanopy.RH_5_cal]
labels_InCanopy = ['0.105 m', '0.29 m', '0.61 m', '0.99 m', '1.49 m']

## Open radionde data (estimations done through the parcel method and eye inspection)
### Load boundary layer characteristics of the radiosondes of Els Plans and La Cendrosa determine by eye inspection
CBL_EP = pd.read_excel("/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/"+\
                                       "PhD_Wageningen/1er-paper/Boundary-"+\
                                           "layer-17-07-2021/CBLcharacteris"+\
                                              "tics_by_eye_EP.xlsx", header = 1)
CBL_LC = pd.read_excel("/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/"+\
                                       "PhD_Wageningen/1er-paper/Boundary-"+\
                                           "layer-17-07-2021/CBLcharacteris"+\
                                              "tics_by_eye_LC.xlsx", header = 1)
CBL_EP = CBL_EP[1:]
CBL_LC = CBL_LC[1:]


### Load boundary layer characteristics of the radiosondes of Els Plans and La Cendrosa determine by the parcel method
CBL_EP_PM = pd.read_csv("/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/"+\
                        "1er-paper/Boundary-layer-17-07-2021/"+\
                            "CBLcharacteristics_by_PM_EP.csv")
CBL_LC_PM = pd.read_csv("/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/"+\
                        "1er-paper/Boundary-layer-17-07-2021/"+\
                            "CBLcharacteristics_by_PM_LC.csv")

## Open Oscar's sensor of PAR
EC_PAR = xr.open_dataset('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/'+\
                         'PhD_Wageningen/1er-paper/observations/'+\
                             'InCanopyProfiles/LIAISE_PARavg_in10min'+\
                                 '_out10min.nc')
### Convert HHMM to string
EC_HHMM = ['%04d'%HHMM_i.item() for HHMM_i in EC_PAR.HHMM]
EC_PAR['datetime'] = [dt.datetime.strptime('17/07/2021 '+HHMM_i, '%d/%m/%Y %H%M')-dt.timedelta(minutes=5) for HHMM_i in EC_HHMM]
EC_PAR = EC_PAR.where(EC_PAR.DOY == 198)

#%% CO2
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



# Load Fabien observations of CO2 at Els Plans (8.5 m)

t_UTC_ElsPlans_Fab = []

#Open the file
with open('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/'+\
          '1er-paper/observations/Fabien_CO2_ElsPlans/t.txt') as fp:
    #Iterate through each line
    for line in fp:

        t_UTC_ElsPlans_Fab.extend( #Append the list of numbers to the result array
            [float(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])

t_UTC_ElsPlans_Fab = np.array(t_UTC_ElsPlans_Fab)

filter_ = (t_UTC_ElsPlans_Fab <23.99999999)

t_UTC_ElsPlans_Fab = t_UTC_ElsPlans_Fab[filter_]


hour = np.floor(t_UTC_ElsPlans_Fab)
minute = np.floor((t_UTC_ElsPlans_Fab-hour)*60)
second = np.floor((t_UTC_ElsPlans_Fab-hour-minute/60)*3600)
datetime_ElsPlans_Fab = [dt.datetime(2021,7,17,int(hour[i]),int(minute[i]),int(second[i])) for i in np.arange(len(t_UTC_ElsPlans_Fab))]

CO2_ElsPlans_Fab = []

#Open the file
with open('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/'+\
          '1er-paper/observations/Fabien_CO2_ElsPlans/xco2.txt') as fp:
    #Iterate through each line
    for line in fp:

        CO2_ElsPlans_Fab.extend( #Append the list of numbers to the result array
            [float(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])

CO2_ElsPlans_Fab = np.array(CO2_ElsPlans_Fab)
CO2_ElsPlans_Fab = CO2_ElsPlans_Fab[filter_]
#%%
# Run baseline model
#r1    = LC_functions.laCendrosa_local_ags_fitted_HugoKevin_adv()
#r1    = LC_final.laCendrosa_local_ags_baseline_advection()
r1 = cases.control()
#r1 = LC_functions.laCendrosa_local_and_advection()
hour_r1 = np.floor(r1.out.t)
minute_r1 = np.floor((r1.out.t-hour_r1)*60)
r1_datetime = [dt.datetime(2021,7,17,int(hour_r1[i]),int(minute_r1[i])) for i in np.arange(len(r1.out.t))]

#%% Constants

P0         = 100000                # Reference pressure [Pa]
Rd_cp      = 0.286                 # Rd/cp
mco2       =  44.;                 # molecular weight CO2 [g mol -1]
mair       =  28.9;                # molecular weight air [g mol -1]
rho        = 1.2                   # density of air [kg m-3]
g          = 9.81                  # gravity acceleration [m/s2]
T_t        = 273.16                # conversion constant from ºC to K

# Conversion of photon flux density at the photosynthetic active range to photosynthetic active radiation

Na         = 6.0221408e23        # Avogadro's number (# of particles in 1 mol)
h          = 6.62607015e-34      # Planck's constant (J*s)
c          = 299792458           # Light speed constant (m/s)
lambda_PAR = 550e-9              # Wavelenght of mean photon in the active photosynthetic range (m)

from_Q_to_PAR = Na*h*c/lambda_PAR*1e-6

#%% Functions needed for the script
def theta_h(T,h):
    theta = T/(1-0.1*h*rho*g/P0)**Rd_cp
    return theta

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

def qsat(T,p):
    return 0.622 * esat(T) / p

#%% General plot settings

cmap = cm.get_cmap('viridis')       # colormap
s = 8                               # size of dots of scatterplot
date_form = DateFormatter("%H")  # Date/Time format
fs = 12

#fig, axs = plt.subplots(2,2, dpi = 300, figsize = (10,10))
fig, axs = plt.subplots(2,2, dpi = 300, figsize = (10,6))

ax1 = axs[0,0]
ax4 = axs[0,1]
ax2 = axs[1,0]
ax3 = axs[1,1]
# gs = gridspec.GridSpec(2, 4, figure = fig)
# gs.update(wspace=0.5)
# ax1 = plt.subplot(gs[0,1:3])
# ax2 = plt.subplot(gs[1,:2])
# ax3 = plt.subplot(gs[1,2:])

index_color = np.linspace(0,1,9)[::-1]

# (1a) PAR 
ax1.scatter(EC_PAR.datetime, (EC_PAR.PAR1+EC_PAR.PAR2)/2, s= s, label ='Sensor at 2 m', color = cmap(index_color[5]), marker = 'o', zorder = 3)

# Plot baseline model
ax1.plot(r1_datetime, r1.out.Swin*0.465/from_Q_to_PAR, label = 'CONTROL', color = 'k')

# Plot a vertical line at solar noon
ax1.axvline(dt.datetime(2021,7,17,12,2,27), zorder = 2, color ='orange', linestyle = '-', linewidth = 0.95)   
ax1.grid()
ax1.set_title("(3a) PAR", fontsize = fs+2)
ax1.set_xlim(dt.datetime(2021,7,17,4),dt.datetime(2021,7,17,20))
ax1.set_xlabel('Time [UTC]', fontsize = fs)
ax1.set_ylabel(r'$\mu$mol $\gamma$ m$^{-2}$ s$^{-1}$', fontsize = fs)
ax1.xaxis.set_major_formatter(date_form)
#ax1.legend(loc = 'best', fontsize = fs)
ax1b = ax1.twinx()

def tick_function_PAR(PAR_micromol):
    PAR_W_m2 = PAR_micromol * from_Q_to_PAR
    return ["%i" % round(z) for z in PAR_W_m2] 
new_tick_locations_PAR = np.array([0,500,1000,1500,2000])

ax1b.set_ylim(ax1.get_ylim())
ax1b.set_yticks(new_tick_locations_PAR)
ax1b.set_yticklabels(tick_function_PAR(new_tick_locations_PAR))
ax1b.set_ylabel(r'W m$^{-2}$', fontsize = fs)

# (1b) Potential temperature

s = 3

ax2.grid()

# Plot tower data
ax2.scatter(data_tower.time_, theta_h(data_tower.ta_2 + T_t, h = 2)-T_t, label = '2 m', s = s, color = cmap(index_color[5]),zorder = 11, alpha = 0.6)
ax2.scatter(data_tower.time_, theta_h(data_tower.ta_3 + T_t, h = 10)-T_t, label = '10 m', s = s, color = cmap(index_color[6]),zorder = 11)
ax2.scatter(data_tower.time_, theta_h(data_tower.ta_4 + T_t, h = 25)-T_t, label = '25 m', s = s, color = cmap(index_color[7]),zorder = 11)
ax2.scatter(data_tower.time_, theta_h(data_tower.ta_5 + T_t, h = 50)-T_t, label = '45 m', s = s, color = cmap(index_color[8]),zorder = 11)

index_color_small_tower = [0,1,2,3,4]
markers_inCanopy = ['o','o','o','o','o']

# Plot small tower (In canopy data)
for i in np.arange(len(T)):
    ax2.scatter(data_InCanopy.DATETIME, theta_h(T[i]+T_t,h = heights[i] )-T_t, label = labels_InCanopy[i], s = s, color = cmap(index_color[index_color_small_tower[i]]), marker = markers_inCanopy[i],zorder = 11)

# Plot values of the mixed layer estimated thanks to the radiosondes
CBL_EP_PM['datetime']= [dt.datetime(2021,7,17,int(np.floor(CBL_EP_PM.Launch_hour_float[i]))) for i in np.arange(len(CBL_EP_PM.Launch_hour_float))]
CBL_LC_PM['datetime']= [dt.datetime(2021,7,17,int(np.floor(CBL_LC_PM.Launch_hour_float[i])), int(np.round((CBL_LC_PM.Launch_hour_float[i]-np.floor(CBL_LC_PM.Launch_hour_float[i]))*60))) for i in np.arange(len(CBL_LC_PM.Launch_hour_float))]
ax2.scatter(CBL_LC_PM["datetime"], CBL_LC_PM["theta"], color = 'k', label = r'$<\theta>$',s = 4, zorder = 11)

# Plot solar noon
ax2.axvline(dt.datetime(2021,7,17,12,2,27), zorder = 2, color ='orange', linestyle = '-', linewidth = 0.95)   

filter_instability = r1.out.t>15.6
theta = r1.out.theta
thetasurf = r1.out.thetasurf
T2m = r1.out.T2m
T0_105m = r1.out.T0_105m
theta[filter_instability] = np.nan
thetasurf[filter_instability] = np.nan
T2m[filter_instability] = np.nan
T0_105m[filter_instability] = np.nan
 
line_T_2m, = ax2.plot(r1_datetime[1:], T2m[1:]-T_t, label = r'CONTROL, $\theta_{surf}$', color = 'k', ls='-', zorder =12)
line_T_0105m, = ax2.plot(r1_datetime[1:], T0_105m[1:]-T_t, label = r'CONTROL, $\theta_{surf}$', color = 'k', ls='--', zorder =12)
line_theta, = ax2.plot(r1_datetime[1:], theta[1:]-T_t, label = r'CONTROL, $\theta_{surf}$', color = 'k', ls=':', zorder =12)

handles, labels = ax2.get_legend_handles_labels()

#specify order of items in legend
order = [4,5,6,7,8,0,1,2,3,9,10]
#add legend to plot
#ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'best',bbox_to_anchor=(1.04,1), title = 'Heights', markerscale = 7)
#ax2.set_ylim(10,40)
ax2.set_title(r'(3c) $\theta$', fontsize = fs+2)
ax2.set_xlim(dt.datetime(2021,7,17,4),dt.datetime(2021,7,17,20))
ax2.set_xlabel('Time [UTC]', fontsize = fs)
ax2.set_ylabel('ºC', fontsize = fs)
ax2.xaxis.set_major_formatter(date_form)

# (1c) Vapor pressure deficit

index_color = np.linspace(0,1,9)[::-1]

# Plot 50 m tower data
ax3.scatter(data_tower.time, esat(data_tower.ta_2 +T_t)-1/0.622*data_tower.rv_1*1e-3*(P0-2*rho*g), label = '3 m', s = s, color = cmap(index_color[5]),zorder = 11)
ax3.scatter(data_tower.time, esat(data_tower.ta_3 +T_t)-1/0.622*data_tower.rv_1*1e-3*(P0-10*rho*g), label = '10 m', s = s, color = cmap(index_color[6]),zorder = 11)
ax3.scatter(data_tower.time, esat(data_tower.ta_4 +T_t)-1/0.622*data_tower.rv_1*1e-3*(P0-25*rho*g), label = '25 m', s = s, color = cmap(index_color[7]),zorder = 11)
ax3.scatter(data_tower.time, esat(data_tower.ta_4 +T_t)-1/0.622*data_tower.rv_1*1e-3*(P0-45*rho*g), label = '45 m', s = s, color = cmap(index_color[8]),zorder = 11)

# Plot small tower data (In canopy data)
index_color_small_tower = [0,1,2,3,4]
for i in np.arange(len(e)):
    ax3.scatter(data_InCanopy.DATETIME, (1-RH[i])*esat(T[i]+T_t), label = labels_InCanopy[i], s =s, color = cmap(index_color[i]), marker = markers_inCanopy[i], zorder = 10)

# Plot radiosonde data
ax3.scatter(CBL_LC_PM.datetime, esat(CBL_LC_PM["theta"]+T_t)-CBL_LC_PM["q"]*1e-3/0.622*(P0-rho*g*50), color = 'k', label = r'$\mathrm{z_{sl}}$-$\mathrm{z_{h}}$ m', s = 2, zorder = 10)

# Plot solar noon
ax3.axvline(dt.datetime(2021,7,17,12,2,27), zorder = 2, color ='orange', linestyle = '-', linewidth = 0.95)    

# Create first legend
##specify order of items in legend
order = [4,5,6,7,8,0,1,2,3,9]
handles, labels = ax3.get_legend_handles_labels()
first_legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower center',bbox_to_anchor=(0.5,-2.35), title = 'Sensor height', markerscale = 7, ncol = 3, fontsize = fs, title_fontsize=14)
ax_legend = plt.gca().add_artist(first_legend)

# Plot baseline model
esat_ = r1.out.esat
esat_[filter_instability] = np.nan
e_ = r1.out.e
e_[filter_instability] = np.nan
e2m = r1.out.e2m
e2m[filter_instability] = np.nan
e0_105m = r1.out.e0_105m
e0_105m[filter_instability] = np.nan
#line_VPD, = ax3.plot(r1_datetime[1:], esat_[1:]-e_[1:], label = r'Baseline, VPD', color = 'k',zorder = 13,linewidth =0.9)
line_VPD_2m, = ax3.plot(r1_datetime[1:], esat(T2m[1:])-e2m[1:], label = r'2 m', color = 'k', ls='-',zorder = 13)
line_VPD_0105m, = ax3.plot(r1_datetime[1:], esat(T0_105m[1:])-e0_105m[1:], label = r'0.105 m', color = 'k', ls='--',zorder = 13)
line_VPD_CBL, = ax3.plot(r1_datetime[1:], esat(theta[1:])-e_[1:], label = r'$\mathrm{z_{sl}}$-$\mathrm{z_{h}}$ m', color = 'k', ls=':', zorder = 13)

# Create second legend
second_legend = plt.legend(handles=[line_VPD_0105m, line_VPD_2m, line_VPD_CBL], loc='best', bbox_to_anchor=(2.3,-1.7), fontsize = fs, title = 'Height of CONTROL simulation', title_fontsize = 14)
ax_legend_2 = plt.gca().add_artist(second_legend)

ax3.set_title('(3d) VPD', fontsize = fs+2)
ax3.set_xlim(dt.datetime(2021,7,17,4),dt.datetime(2021,7,17,20))
ax3.set_xlabel('Time [UTC]', fontsize = fs)
ax3.set_ylabel(r'Pa', fontsize = fs)
ax3.xaxis.set_major_formatter(date_form)
ax3.grid()

# (1b) Atmospheric CO2

# Plot solar noon
ax4.axvline(dt.datetime(2021,7,17,12,2,27), zorder = 2, color ='orange', linestyle = '-', linewidth = 0.95)    

ax4.plot(r1_datetime[1:], r1.out.CO2[1:], c = 'k', label = 'CONTROL', linestyle = ':', zorder = 10)
ax4.scatter(EC_data.datetime, CO2_EC, label = 'Sensor at 2.45 m',s=10, color = cmap(index_color[5]), zorder = 5)
#ax4.scatter(datetime_ElsPlans_Fab, CO2_ElsPlans_Fab, marker = '<', s = 10, color = 'grey',label = 'Els Plans, 8.5 m', zorder = 5)
ax4.set_title('(3b) Atmospheric CO$_2$', fontsize = fs+2)
ax4.set_xlim(dt.datetime(2021,7,17,4),dt.datetime(2021,7,17,20))
ax4.grid()
ax4.set_xlabel('Time [UTC]', fontsize = fs)
ax4.set_ylabel(r'ppm', fontsize = fs)
ax4.xaxis.set_major_formatter(date_form)
#ax4.legend(loc = 'best', fontsize = fs)


plt.tight_layout()
plt.show()

