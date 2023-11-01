#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:01:25 2023

@author: raquelgonzalezarmas
"""


"""
    ===> Create figure of main forcings that have been modified in the three 
    scenarios
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as final_cases

#%% Needed functions and cosntants
def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

T_0 = 273.15
#%% Run simulations
r1    = final_cases.control()
r2    = final_cases.control_cloud()
r3    = final_cases.control_ent_DA()
r4    = final_cases.control_adv_CA()

hour_r1 = np.floor(r1.out.t)
minute_r1 = np.floor((r1.out.t-hour_r1)*60)
r1_datetime = [dt.datetime(2021,7,17,int(hour_r1[i]),int(minute_r1[i])) for i in np.arange(len(r1.out.t))]

simulations = [r1 ,r2 ,r3 ,r4]
labels = ['Control', 'PAR-CLD', 'VPD-ENT', 'TEM-ADV']
linestyles = ['-', '--', ':', '--']
markers = ['']
time_max = 12
fs = 12
filter_ = (r1.out.t< 15.5)
x_axis_ticks = np.arange(4,22,2)

#%% Create figures of the three forcings implemented in each scenario
fig, axs = plt.subplots(1, 3, dpi = 350, figsize = (12,3))

# Plot stomatal conductance
ax1, ax2, ax3 = axs[0], axs[1], axs[2]

# (1) Scenario 1. Change in radiation due to a cloud shade.
ax1.plot(r1.out.t[filter_],r1.out.PAR[filter_], label = 'Control', c = 'k')
ax1.plot(r2.out.t[filter_],r2.out.PAR[filter_], label = 'PAR-CLD', c = 'k', linestyle ='--')
ax1.set_xlabel('Time [UTC]')
ax1.set_ylabel(r'PAR [W m$^{-2}$]')
ax1.set_title('(2a) PAR-CLD vs Control')
ax1.set_xlim(4,20)
ax1.set_xticks(x_axis_ticks)
ax1.grid()
ax1.legend()


# (2) Scenario 2. Change in free tropospheric water content. An initial dryier free troposphere.
dqdt_r1 = (r1.out.wq-r1.out.wqe-r1.out.wqM)/r1.out.h + r1.out.advq
dqdt_r3  = (r3.out.wq-r3.out.wqe-r3.out.wqM)/r3.out.h + r3.out.advq


ax2b = ax2.twinx()
ax2.plot(r1.out.t[filter_],r1.out.q[filter_]*1e3+r1.out.dq[filter_]*1e3, label = 'Control', c = 'k')
ax2.plot(r3.out.t[filter_],r3.out.q[filter_]*1e3+r3.out.dq[filter_]*1e3, label = 'VPD-ENT', c = 'k', linestyle = ':')
# ax2b.plot(r1.out.t[filter_],r1.out.wqe[filter_]/(r1.out.wq[filter_]+r1.out.wqe[filter_]), label = 'Control', c = 'blue')
# ax2b.plot(r3.out.t[filter_],r3.out.wqe[filter_]/(r3.out.wq[filter_]+r3.out.wqe[filter_]), label = 'Ent. D.A.', c = 'blue', linestyle = ':')
ax2b.plot(r1.out.t[filter_],r1.out.wqe[filter_]*1e3, label = 'Control', c = 'blue')
ax2b.plot(r3.out.t[filter_],r3.out.wqe[filter_]*1e3, label = 'VPD-ENT', c = 'blue', linestyle = ':')
ax2.set_xlabel('Time [UTC]')
ax2.set_ylabel(r'q$_{FT}$(h=h$_{CBL}$) [g$_{water}$ kg$^{-1}_{air}$]')
ax2b.set_ylabel(r"$\left(\overline{w'q'}\right)_{e}$ [g$_{water}$ kg$^{-1}_{air}$ m s$^{-1}$]")
ax2b.spines['right'].set_color('blue')
ax2b.yaxis.label.set_color('blue')
ax2b.tick_params(axis='y', colors='blue')
ax2.grid()
ax2.legend(loc = 'lower center')
ax2.set_title('(2b) VPD-ENT vs Control')
ax2.set_xlim(4,20)
ax2.set_xticks(x_axis_ticks)

# (3) Scenario 3. Change in advection of air. Advection of cold air during the afternoon.
ax3.plot(r1.out.t[filter_],r1.out.advtheta[filter_]*3600, label = 'Control', c = 'k')
ax3.plot(r4.out.t[filter_],r4.out.advtheta[filter_]*3600, label = 'TEM-ADV', c = 'k', linestyle = '-.')
ax3.set_xlabel('Time [UTC]')
ax3.set_ylabel(r'adv$_{\theta}$ [K h$^{-1}$]')
ax3.grid()
ax3.legend()
ax3.set_title('(2c) TEM-ADV vs Control')
ax3.set_xlim(4,20)
ax3.set_xticks(x_axis_ticks)

ax1.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
ax2.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
ax3.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')

fig.tight_layout()

#%% Create a figure with the 3 environmental factors#%% Create figure of the 4 environmental drivers
fig, axs = plt.subplots(1, 3, dpi = 350, figsize = (10,3))

# Plot stomatal conductance
ax2, ax3, ax4 = axs[0], axs[1], axs[2]

# (b) CO2
ax2.plot(r1.out.t[filter_],r1.out.CO2_0_105m[filter_], label = 'Control', c = 'k')
ax2.plot(r2.out.t[filter_],r2.out.CO2_0_105m[filter_], label = 'PAR-CLD', c = 'k', linestyle ='--')
ax2.plot(r3.out.t[filter_],r3.out.CO2_0_105m[filter_], label = 'VPD-ENT', c = 'k', linestyle =':')
ax2.plot(r4.out.t[filter_],r4.out.CO2_0_105m[filter_], label = 'TEM-ADV', c = 'k', linestyle ='-.')
ax2.set_xlabel('Time [UTC]')
ax2.set_ylabel(r'ppm')
ax2.grid()
ax2.legend()
ax2.set_title('(7a) Atmospheric CO$_2$')
ax2.set_xlim(4,20)
ax2.set_xticks(x_axis_ticks)

# (c) T
ax3.plot(r1.out.t[filter_],r1.out.T0_105m[filter_]-T_0, label = 'Control', c = 'k')
ax3.plot(r2.out.t[filter_],r2.out.T0_105m[filter_]-T_0, label = 'PAR-CLD', c = 'k', linestyle ='--')
ax3.plot(r3.out.t[filter_],r3.out.T0_105m[filter_]-T_0, label = 'VPD-ENT', c = 'k', linestyle =':')
ax3.plot(r4.out.t[filter_],r4.out.T0_105m[filter_]-T_0, label = 'TEM-ADV', c = 'k', linestyle ='-.')
ax3.set_xlim(4,20)
ax3.set_xlabel('Time [UTC]')
ax3.set_ylabel(r'ºC')
ax3.set_title(r'(7b) Potential T')
ax3.grid()
ax3.set_xlim(4,20)
ax3.set_xticks(x_axis_ticks)

# (d) VPD
ax4.plot(r1.out.t[filter_],esat(r1.out.T0_105m[filter_])-r1.out.e0_105m[filter_], label = 'Control', c = 'k')
ax4.plot(r2.out.t[filter_],esat(r2.out.T0_105m[filter_])-r2.out.e0_105m[filter_], label = 'PAR-CLD', c = 'k', linestyle ='--')
ax4.plot(r3.out.t[filter_],esat(r3.out.T0_105m[filter_])-r3.out.e0_105m[filter_], label = 'VPD-ENT', c = 'k', linestyle =':')
ax4.plot(r4.out.t[filter_],esat(r4.out.T0_105m[filter_])-r4.out.e0_105m[filter_], label = 'TEM-ADV', c = 'k', linestyle ='-.')
ax4.set_xlim(4,20)
ax4.set_xlabel('Time [UTC]')
ax4.set_ylabel(r'Pa')
ax4.grid()
ax4.set_title('(7c) VPD')
ax4.set_xlim(4,20)
ax4.set_xticks(x_axis_ticks)

ax2.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
ax3.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
ax4.axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')


fig.tight_layout()