#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:06:14 2023

@author: raquelgonzalezarmas
"""

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
                'PhD_Wageningen/1er-paper/models/Tendencies/')
import calculate_tendencies_function as tendencies
import numpy as np
import matplotlib.gridspec as gridspec
#%% Run case
r1 = cases.control()

tendencies_all, tendencies = tendencies.calculate_tendencies(r1)

#%% Trouble shooting tendencies


#%% Figure final panel
fig = plt.figure(dpi = 350, figsize = (10,6))
gs = gridspec.GridSpec(2, 2, figure = fig)
gs.update(wspace=0.55)

ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0,1:2])
ax3 = plt.subplot(gs[1,0:1])
fs = 12

t = r1.out.t
filter_ = (t>5.25)&(t<15.5)

Lv         = 2.5e6                 # heat of vaporization [J kg-1]
mco2       =  44.;                 # molecular weight CO2 [g mol -1]
mair       =  28.9;                # molecular weight air [g mol -1]
mh2o       = 18.;                  # molecular weight water [g mol-1]
nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide 
cte = 0.0248



# Stomatal conductance to water vapor
# Total tendencies
ax1.plot(t[filter_], tendencies.dgswdt[filter_], c = "k", label = 'Total', zorder = 15)
#ax1.plot(t[filter_],  tendencies.dgswdt_PAR[filter_]+ tendencies.dgswdt_Ds[filter_]+ \
 #           tendencies.dgswdt_T[filter_]+ tendencies.dgswdt_Cs[filter_],c = 'k', linestyle = '--')
# Partial tendencies

ax1.plot(t[filter_], tendencies.dgswdt_PAR[filter_], \
        label = "Radiation", c = 'orange', zorder = 14)
ax1.plot(t[filter_], tendencies.dgswdt_Ds[filter_], \
        label = r"Vapor pressure deficit", c = 'blue', zorder = 13)
ax1.plot(t[filter_], tendencies.dgswdt_T[filter_], \
        label = r"Temperature", c = 'red', zorder = 12)
ax1.plot(t[filter_], tendencies.dgswdt_Cs[filter_], \
     label = r"Air CO$_2$", c = 'green', zorder = 11)
#ax1.plot(x[filter_], (dglwdw2*dw2dt)[filter_], \
 #       label = r"Soil water content", c = 'brown')
ax1.grid()
ax1.set_xlim(4, 20)
ax1.set_ylim(-0.01/3600, 0.01/3600)
ax1.set_xlabel('Time [UTC]', fontsize = fs)
ax1.set_ylabel(r'$\mathrm{m~s^{-2}}$', fontsize = fs)
ax1.set_title('(6a) Tendencies of $g_{s,w}$', fontsize = fs)
ax1.tick_params(axis='both', labelsize = fs)

## add second axis
axs0b = ax1.twinx()

def tick_function_dgsdt(dgsdt):
    dgsdt_new = dgsdt / cte
    return ["%.0e" %z for z in dgsdt_new] 

new_tick_locations_dgsdt = np.array([-2,-1,0,1,2])*1e-6 
axs0b.set_ylim(ax1.get_ylim())
axs0b.set_yticks(new_tick_locations_dgsdt)
axs0b.set_yticklabels(tick_function_dgsdt(new_tick_locations_dgsdt), fontsize = fs)
axs0b.set_ylabel(r'mol$_{air}~m_{leaf}^{-2}~s^{-2}$', fontsize = fs)


# Transpiration rate
# Total tendencies
ax3.plot(t[filter_], tendencies.dTRdt[filter_]*1e6, c = "k", label = 'Total', zorder = 15)
#ax3.plot(t[filter_],  (tendencies.dTRdt_PAR[filter_]+ tendencies.dTRdt_Ds[filter_]+ \
 #           tendencies.dTRdt_T[filter_]+ tendencies.dTRdt_Cs[filter_])*1e6,c = 'k', linestyle = '--')
# Partial tendencies

ax3.plot(t[filter_], tendencies.dTRdt_PAR[filter_]*1e6, \
        label = "Radiation", c = 'orange', zorder = 14)
ax3.plot(t[filter_], tendencies.dTRdt_Ds[filter_]*1e6, \
        label = r"Vapor pressure deficit", c = 'blue', zorder = 13)
ax3.plot(t[filter_], tendencies.dTRdt_T[filter_]*1e6, \
        label = r"Temperature", c = 'red', zorder = 12)
ax3.plot(t[filter_], tendencies.dTRdt_Cs[filter_]*1e6, \
        label = r"Air CO$_2$", c = 'green', zorder = 11)
#ax3.plot(x[filter_], (dTRdw2*dw2dt)[filter_]*1e6, \
 #       label = r"Soil water content", c = 'brown')
#ax3.legend(bbox_to_anchor=(3, 0.5), fontsize = fs, ncol = 1)
ax3.grid()
ax3.set_ylim(-0.02, 0.02)
ax3.set_xlabel('Time [UTC]', fontsize = fs)
ax3.set_ylabel(r'$\mathrm{mg_{water}~m^{-2}~s^{-2}}$', fontsize = fs)
ax3.set_title('(6c) Tendencies of TR$_{leaf}$', fontsize = fs)
ax3.set_xlim(4, 20)
ax3.tick_params(axis='both', labelsize = fs)

## add second axis
axs2b = ax3.twinx()

def tick_function_dTRdt(dTRdt):
    dTRdt_new = dTRdt / mh2o
    return ["%.0e" %z for z in dTRdt_new] 

new_tick_locations_dTRdt = np.array([-0.02,-0.01, 0 ,0.01, 0.02]) 

axs2b.set_ylim(ax3.get_ylim())
axs2b.set_yticks(new_tick_locations_dTRdt)
axs2b.set_yticklabels(tick_function_dTRdt(new_tick_locations_dTRdt), fontsize = fs)
axs2b.set_ylabel(r'mmol$_{H_2O}$ m$_{leaf}^{-2}$  s$^{-1}$', fontsize = fs)

# Net assimilation rate
# Total tendencies
ax2.plot(t[filter_], tendencies.dAndt[filter_], c = "k", label = r'$\frac{d A_{nl}}{dt}$', zorder = 15)
#ax2.plot(t[filter_],  tendencies.dAndt_PAR[filter_]+ tendencies.dAndt_Ds[filter_]+ \
 #           tendencies.dAndt_T[filter_]+ tendencies.dAndt_Cs[filter_],c = 'k', linestyle = '--')
# Partial tendencies
ax2.plot(t[filter_], tendencies.dAndt_PAR[filter_], \
        label = r"$\frac{\partial A_{nl}}{\partial PAR}\cdot \frac{dPAR}{dt}$", c = 'orange',  zorder = 14)
ax2.plot(t[filter_], tendencies.dAndt_Cs[filter_], \
        label = r"$\frac{\partial A_{nl}}{\partial C_s}\cdot \frac{dC_s}{dt}$", c = 'green', zorder = 11)
ax2.plot(t[filter_], tendencies.dAndt_Ds[filter_], \
        label = r"$\left(\frac{\partial A_{nl}}{\partial D_s}\right)_{T}\cdot \frac{dD_s}{dt}$", c = 'blue', zorder = 13)
ax2.plot(t[filter_], tendencies.dAndt_T[filter_], \
        label = r"$\left(\frac{\partial A_{nl}}{\partial T}\right)_{D_s}\cdot \frac{dT}{dt}$", c = 'red', zorder = 12)
# ax3.plot(x[filter_], (dAnldw2*dw2dt)[filter_], \
 #       label = r"$\frac{\partial A_{nl}}{\partial w_2}\cdot \frac{dw_2}{dt}$", c = 'brown')
ax2.grid()
ax2.set_ylim(-0.65/3600, 0.65/3600)
ax2.set_xlabel('Time [UTC]', fontsize = fs)
ax2.set_ylabel(r'$\mathrm{mg_{CO_2}~m^{-2}~s^{-2}}$', fontsize = fs)
ax2.set_title('(6b) Tendencies of $A_{n}$', fontsize = fs)
ax2.set_xlim(4, 20)
ax2.tick_params(axis='both', labelsize = fs)

## add second axis
axs1b = ax2.twinx()

def tick_function_dAndt(dAndt):
    dAndt_new = dAndt / mco2 *1e3
    return ["%.0e" %z for z in dAndt_new] 

new_tick_locations_dgsdt = np.array([-1e-4,0,1e-4])
axs1b.set_ylim(ax2.get_ylim())
axs1b.set_yticks(new_tick_locations_dgsdt)
axs1b.set_yticklabels(tick_function_dgsdt(new_tick_locations_dgsdt), fontsize = fs)
axs1b.set_ylabel(r'mol$_{air}~m_{leaf}^{-2}~s^{-2}$', fontsize = fs)



ax1.axvline(12, linewidth = 0.95, zorder = 2, color = 'k')
ax2.axvline(12, linewidth = 0.95, zorder = 2, color = 'k')
ax3.axvline(12, linewidth = 0.95, zorder = 2, color = 'k')

ax1.axhline(0, linewidth = 0.95, zorder = 2, color = 'k')
ax2.axhline(0, linewidth = 0.95, zorder = 2, color = 'k')
ax3.axhline(0, linewidth = 0.95, zorder = 2, color = 'k')

gs.tight_layout(fig)

#%%


fig =  plt.figure(dpi = 350)
figlegend = plt.figure(dpi = 350, figsize=(3,2))
ax = fig.add_subplot(111)
# Total tendencies
line1, = ax.plot(t[filter_], tendencies.dgswdt[filter_], c = "k", label = 'Total', zorder = 15)

# Partial tendencies

line2, = ax.plot(t[filter_], tendencies.dgswdt_PAR[filter_], \
        label = "Radiation", c = 'orange', zorder = 14)
line3, = ax.plot(t[filter_], tendencies.dgswdt_Ds[filter_], \
        label = r"Vapor pressure deficit", c = 'blue', zorder = 13)
line4, = ax.plot(t[filter_], tendencies.dgswdt_T[filter_], \
        label = r"Temperature", c = 'red', zorder = 12)
line5, = ax.plot(t[filter_], tendencies.dgswdt_Cs[filter_], \
     label = r"Air CO$_2$", c = 'green', zorder = 11)

figlegend.legend([line1, line2, line3, line4, line5],('Total', 'Radiation', 'VPD', 'Temperature', r'Air CO$_2$'), loc ='center')
fig.show()
figlegend.show()
figlegend.savefig('legend.png')
