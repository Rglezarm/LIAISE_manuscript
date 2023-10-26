#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:40:16 2023

@author: raquelgonzalezarmas
"""


"""
    ===> Script of figure 5 but using the function of calculating tendencies
"""


import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/'+\
                'PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as cases
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/'+\
                'PhD_Wageningen/1er-paper/models/Repository_BG_manuscript/tendencies_leaf_gas_exchange/')
import calculate_tendencies_function as tendencies
import numpy as np
from matplotlib.ticker import ScalarFormatter
#%% Run case
r1 = cases.control()
#r1 = cases.control_cloud()
#r1 = cases.control_ent_DA() 
#r1 = cases.control_adv_CA()

tendencies_all, tendencies = tendencies.calculate_tendencies(r1)


#%% Figure final panel
fig, axs = plt.subplots(1,3, dpi = 300, figsize = (12,4))
fs = 12

# Create the legend as a separate figure
figLegend = plt.figure(figsize = (7,1), dpi = 350)

t = r1.out.t
filter_ = (t>5.25)&(t<15.5)

Lv         = 2.5e6                 # heat of vaporization [J kg-1]
mco2       =  44.;                 # molecular weight CO2 [g mol -1]
mair       =  28.9;                # molecular weight air [g mol -1]
mh2o       = 18.;                  # molecular weight water [g mol-1]
nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide 
cte        = 0.0248



# Stomatal conductance to water vapor
# Total tendencies
line_total_gsw, = axs[0].plot(t[filter_], (tendencies.dgswdt[filter_])/cte, c = "k", label = 'Total', zorder = 15)
line_summed_gsw, = axs[0].plot(t[filter_], (tendencies.dgswdt_PAR[filter_]+ tendencies.dgswdt_Ds[filter_]+ \
            tendencies.dgswdt_T[filter_]+ tendencies.dgswdt_Cs[filter_])/cte,c = 'lightgrey', linestyle = '--', zorder = 16)
# Partial tendencies

line_PAR_gsw, = axs[0].plot(t[filter_], tendencies.dgswdt_PAR[filter_]/cte, \
        label = "Radiation", c = 'orange', zorder = 14)
line_VPD_gsw, = axs[0].plot(t[filter_], tendencies.dgswdt_Ds[filter_]/cte, \
        label = r"Vapor pressure deficit", c = 'blue', zorder = 13)
line_T_gsw,   = axs[0].plot(t[filter_], tendencies.dgswdt_T[filter_]/cte, \
        label = r"Temperature", c = 'red', zorder = 12)
line_Cs_gsw,  = axs[0].plot(t[filter_], tendencies.dgswdt_Cs[filter_]/cte, \
     label = r"Air CO$_2$", c = 'green', zorder = 11)
#axs[0].plot(x[filter_], (dglwdw2*dw2dt)[filter_], \
 #       label = r"Soil water content", c = 'brown')
axs[0].grid()
axs[0].set_xlim(4, 20)
axs[0].set_ylim(-1e-2/3600/cte, 1e-2/3600/cte)
axs[0].set_xlabel('Time [UTC]', fontsize = fs)
axs[0].set_ylabel(r'$\mathrm{\mu mol~ m_{leaf}^{-2}~s^{-2}}$', fontsize = fs)
axs[0].set_title('(6a) Temporal tendencies of $g_{s}$', fontsize = fs, y=1.08)
axs[0].tick_params(axis='both', labelsize = fs)
axs[0].ticklabel_format(axis='y', style='sci')
"""
- Only for tendencies of the experiments -
y = np.arange(-0.01/3600, 0.02/3600,0.01/3600)
axs[0].fill_betweenx(y,11 + 15/60,15+35/60, alpha = 0.3)
axs[0].annotate('TEM-ADV', xy = (17,0.008/3600), ha='center',fontsize = fs)
"""
"""
## add second axis
axs0b = axs[0].twinx()

def tick_function_dgsdt(dgsdt):
    dgsdt_new = dgsdt / cte
    return ["%.0e" %z for z in dgsdt_new] 

new_tick_locations_dgsdt = np.array([-2,-1,0,1,2])*1e-6 
axs0b.set_ylim(axs[0].get_ylim())
axs0b.set_yticks(new_tick_locations_dgsdt)
axs0b.set_yticklabels(tick_function_dgsdt(new_tick_locations_dgsdt), fontsize = fs)
axs0b.set_ylabel(r'mol$_{air}~m_{leaf}^{-2}~s^{-2}$', fontsize = fs)
axs0b.yaxis.set_major_formatter(ScalarFormatter())
axs0b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
"""

# Transpiration rate
# Total tendencies
axs[2].plot(t[filter_], tendencies.dTRdt[filter_]*1e6, c = "k", label = 'Total', zorder = 15)
axs[2].plot(t[filter_],  (tendencies.dTRdt_PAR[filter_]+ tendencies.dTRdt_Ds[filter_]+ \
            tendencies.dTRdt_T[filter_]+ tendencies.dTRdt_Cs[filter_])*1e6,c = 'lightgrey', linestyle = '--', zorder = 16)
# Partial tendencies

axs[2].plot(t[filter_], tendencies.dTRdt_PAR[filter_]*1e6, \
        label = "Radiation", c = 'orange', zorder = 14)
axs[2].plot(t[filter_], tendencies.dTRdt_Cs[filter_]*1e6, \
        label = r"Air CO$_2$", c = 'green', zorder = 11)
axs[2].plot(t[filter_], tendencies.dTRdt_Ds[filter_]*1e6, \
        label = r"Vapor pressure deficit", c = 'blue', zorder = 13)
axs[2].plot(t[filter_], tendencies.dTRdt_T[filter_]*1e6, \
        label = r"Temperature", c = 'red', zorder = 12)
#axs[1].plot(x[filter_], (dTRdw2*dw2dt)[filter_]*1e6, \
 #       label = r"Soil water content", c = 'brown')
axs[2].grid()
axs[2].set_ylim(-0.02, 0.02)
axs[2].set_xlabel('Time [UTC]', fontsize = fs)
axs[2].set_ylabel(r'$\mathrm{mg_{water}~m^{-2}~s^{-2}}$', fontsize = fs)
axs[2].set_title('(6c) Temporal tendencies of TR$_{leaf}$', fontsize = fs, y=1.08)
axs[2].set_xlim(4, 20)
axs[2].tick_params(axis='both', labelsize = fs)
"""
- Only for tendencies of the experiments -
y = np.arange(-0.02, 0.03,0.01)
axs[2].fill_betweenx(y,11 + 15/60,15+35/60, alpha = 0.3)
axs[2].annotate('TEM-ADV', xy = (17,0.0165), ha='center',fontsize = fs)
"""

"""
## add second axis
axs2b = axs[2].twinx()

def tick_function_dTRdt(dTRdt):
    dTRdt_new = dTRdt / mh2o
    return ["%.0e" %z for z in dTRdt_new] 

new_tick_locations_dTRdt = np.array([-0.02,-0.01, 0 ,0.01, 0.02]) 

axs2b.set_ylim(axs[2].get_ylim())
axs2b.set_yticks(new_tick_locations_dTRdt)
axs2b.set_yticklabels(tick_function_dTRdt(new_tick_locations_dTRdt), fontsize = fs)
axs2b.set_ylabel(r'mmol$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$', fontsize = fs)
axs2b.yaxis.set_major_formatter(ScalarFormatter())
axs2b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
"""

# Net assimilation rate
# Total tendencies
axs[1].plot(t[filter_], tendencies.dAndt[filter_], c = "k", label = r'Total', zorder = 15)
axs[1].plot(t[filter_],  tendencies.dAndt_PAR[filter_]+ tendencies.dAndt_Ds[filter_]+ \
            tendencies.dAndt_T[filter_]+ tendencies.dAndt_Cs[filter_],c = 'lightgrey', linestyle = '--', zorder = 16)
# Partial tendencies
axs[1].plot(t[filter_], tendencies.dAndt_PAR[filter_], \
        label = r"PAR", c = 'orange',  zorder = 14)
axs[1].plot(t[filter_], tendencies.dAndt_Ds[filter_], \
        label = r"VPD", c = 'blue', zorder = 13)
axs[1].plot(t[filter_], tendencies.dAndt_T[filter_], \
        label = r"T", c = 'red', zorder = 12)
axs[1].plot(t[filter_], tendencies.dAndt_Cs[filter_], \
        label = r"C$_S$", c = 'green', zorder = 11)
# axs[2].plot(x[filter_], (dAnldw2*dw2dt)[filter_], \
 #       label = r"$\frac{\partial A_{nl}}{\partial w_2}\cdot \frac{dw_2}{dt}$", c = 'brown')
axs[1].grid()
axs[1].set_ylim(-0.65/3600, 0.65/3600)
axs[1].set_xlabel('Time [UTC]', fontsize = fs)
axs[1].set_ylabel(r'$\mathrm{mg_{CO_2}~m^{-2}~s^{-2}}$', fontsize = fs)
axs[1].set_title('(6b) Temporal tendencies of $A_{n}$', fontsize = fs, y=1.08)
axs[1].set_xlim(4, 20)
axs[1].tick_params(axis='both', labelsize = fs)
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
"""
- Only for tendencies of the experiments -
y = np.arange(-0.65/3600, 0.66/3600,0.01/3600)
axs[1].fill_betweenx(y,11 + 15/60,15+35/60, alpha = 0.3)
axs[1].annotate('TEM-ADV', xy = (17,0.53/3600), ha='center',fontsize = fs)
"""
"""
## add second axis
axs1b = axs[1].twinx()

def tick_function_dAndt(dAndt):
    dAndt_new = dAndt / mco2 *1e3
    return ["%.0e" %z for z in dAndt_new] 

new_tick_locations_dgsdt = np.array([-1.5, -1, -0.5,0,0.5,1,1.5])*1e-4
axs1b.set_ylim(axs[1].get_ylim())
axs1b.set_yticks(new_tick_locations_dgsdt)
axs1b.set_yticklabels(tick_function_dgsdt(new_tick_locations_dgsdt), fontsize = fs)
axs1b.set_ylabel(r'mol$_{air}~m_{leaf}^{-2}~s^{-2}$', fontsize = fs)
axs1b.yaxis.set_major_formatter(ScalarFormatter())
axs1b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
"""

axs[0].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[1].axvline(12+2/60+27/3600, linewidth = 1, zorder = 2, color = 'orange', linestyle = '--')
axs[2].axvline(12+2/60+27/3600, linewidth = 1, zorder = 2, color = 'orange', linestyle = '--')

axs[0].axhline(0, linewidth = 0.95, zorder = 2, color = 'k')
axs[1].axhline(0, linewidth = 0.95, zorder = 2, color = 'k')
axs[2].axhline(0, linewidth = 0.95, zorder = 2, color = 'k')

x_axis_ticks = np.arange(4,22,2)
axs[0].set_xticks(x_axis_ticks)
axs[1].set_xticks(x_axis_ticks)
axs[2].set_xticks(x_axis_ticks)

fig.tight_layout()

figLegend = plt.figure(figsize = (11,1),dpi = 350)
figLegend.legend([line_total_gsw, line_summed_gsw, line_PAR_gsw, line_VPD_gsw, line_T_gsw, line_Cs_gsw], ["Total term", "Sum of partial terms", "PAR term", "VPD term", "T term", r"C$_a$ term"], ncol = 6, loc = "center")
figLegend.savefig('Fig6_tendencies_legend.png')







