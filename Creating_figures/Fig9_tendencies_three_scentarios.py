#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:21:25 2023

@author: raquelgonzalezarmas
"""

"""
    ===> File that reproduces Figure 9 of LIAISE paper
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as cases
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/'+\
                'PhD_Wageningen/1er-paper/models/Repository_BG_manuscript/tendencies_leaf_gas_exchange/')
import calculate_tendencies_function as tendencies

#%% Run case
r0 = cases.control()
r1 = cases.control_cloud()
r2 = cases.control_ent_DA()
r3 = cases.control_adv_CA()

tendencies_all_r0, tendencies_r0 = tendencies.calculate_tendencies(r0)
tendencies_all_r1, tendencies_r1 = tendencies.calculate_tendencies(r1)
tendencies_all_r2, tendencies_r2 = tendencies.calculate_tendencies(r2)
tendencies_all_r3, tendencies_r3 = tendencies.calculate_tendencies(r3)

simulations = [r0, r1]
tendencies_list = [tendencies_r0, tendencies_r1, tendencies_r2, tendencies_r3]

labels = ['Control', 'PAR-CLD', 'VPD-ENT', 'TEM-ADV']
t = r1.out.t
filter_ = (t>5.25)&(t<15.5)
linestyles = ['-', '--', ':', '-.']

#%% Prototype 1 of Fig. 9
fig, axs = plt.subplots(2, 4, dpi = 350, figsize = (12,6))
fs = 11
filter_ = (t>5.25)&(t<15.5)
Lv         = 2.5e6                 # heat of vaporization [J kg-1]

x_axis_ticks = np.arange(4,22,2)

# Net assimilation
axs[0,0].set_title('(9a)', fontsize = fs+2)
line_An_CONTROL, = axs[0,0].plot(r0.out.t[filter_], r0.out.Aln[filter_], label = 'Control', c ='k')
line_An_PAR_CLD, = axs[0,0].plot(r1.out.t[filter_], r1.out.Aln[filter_], label = 'PAR-CLD', c ='k', linestyle ='--')
line_An_VPD_ENT, = axs[0,0].plot(r1.out.t[filter_], r2.out.Aln[filter_], label = 'VPD-ENT', c ='k', linestyle =':')
line_An_TEM_ADV, = axs[0,0].plot(r1.out.t[filter_], r3.out.Aln[filter_], label = 'TEM-ADV', c ='k', linestyle ='-.')
axs[0,0].set_xlabel('Time [UTC]', fontsize = fs)
axs[0,0].set_ylabel('A$_n$ [mg$_{CO_2}$ m$^{-2}$ s$^{-1}$]', fontsize = fs)
axs[0,0].set_xlim(4,20)
axs[0,0].set_xticks(x_axis_ticks)
axs[0,0].grid()

# Difference of tendencies
axs[0,1].set_title('(9b)', fontsize = fs+2)
line_tend_net, = axs[0,1].plot(r0.out.t[filter_], (tendencies_list[1].dAndt-tendencies_list[0].dAndt)[filter_], c ='k')
line_tend_PAR, = axs[0,1].plot(r0.out.t[filter_], (tendencies_list[1].dAndt_PAR-tendencies_list[0].dAndt_PAR)[filter_], c ='orange')
line_tend_VPD, = axs[0,1].plot(r0.out.t[filter_], (tendencies_list[1].dAndt_Ds-tendencies_list[0].dAndt_Ds)[filter_], c ='blue')
line_tend_TEM, = axs[0,1].plot(r0.out.t[filter_], (tendencies_list[1].dAndt_T-tendencies_list[0].dAndt_T)[filter_], c ='red')
line_tend_Cs , = axs[0,1].plot(r0.out.t[filter_], (tendencies_list[1].dAndt_Cs-tendencies_list[0].dAndt_Cs)[filter_], c ='green')
y = np.arange(-6e-5,7e-5,1e-5)
axs[0,1].fill_betweenx(y,10,14, alpha = 0.3)
axs[0,1].set_xlabel('Time [UTC]', fontsize = fs)
axs[0,1].set_ylabel('$\Delta$ Tendencies A$_n$ [mg$_{CO_2}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[0,1].set_xlim(4,20)
axs[0,1].set_xticks(x_axis_ticks)
axs[0,1].set_ylim(-6e-5,6e-5)
axs[0,1].grid()
axs[0,1].annotate('PAR-CLD' , xy = (8,5*1e-5,), ha='center', fontsize = fs)

axs[0,2].set_title('(9c)', fontsize = fs+2)
axs[0,2].plot(r0.out.t[filter_], (tendencies_list[2].dAndt-tendencies_list[0].dAndt)[filter_], c ='k')
axs[0,2].plot(r0.out.t[filter_], (tendencies_list[2].dAndt_PAR-tendencies_list[0].dAndt_PAR)[filter_], c ='orange')
axs[0,2].plot(r0.out.t[filter_], (tendencies_list[2].dAndt_Ds-tendencies_list[0].dAndt_Ds)[filter_], c ='blue')
axs[0,2].plot(r0.out.t[filter_], (tendencies_list[2].dAndt_T-tendencies_list[0].dAndt_T)[filter_], c ='red')
axs[0,2].plot(r0.out.t[filter_], (tendencies_list[2].dAndt_Cs-tendencies_list[0].dAndt_Cs)[filter_], c ='green')
y = np.arange(-6e-6,7e-6,1e-6)
axs[0,2].fill_betweenx(y,6 + 20/60,15+35/60, alpha = 0.3)
axs[0,2].set_xlabel('Time [UTC]', fontsize = fs)
axs[0,2].set_ylabel('$\Delta$ Tendencies A$_n$ [mg$_{CO_2}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[0,2].set_xlim(4,20)
axs[0,2].set_xticks(x_axis_ticks)
axs[0,2].set_ylim(-6e-6,6e-6)
axs[0,2].grid()
axs[0,2].annotate('VPD-ENT' , xy = (8,5*1e-6,), ha='center', fontsize = fs)

axs[0,3].set_title('(9d)', fontsize = fs+2)
axs[0,3].plot(r0.out.t[filter_], (tendencies_list[3].dAndt-tendencies_list[0].dAndt)[filter_], c ='k')
axs[0,3].plot(r0.out.t[filter_], (tendencies_list[3].dAndt_PAR-tendencies_list[0].dAndt_PAR)[filter_], c ='orange')
axs[0,3].plot(r0.out.t[filter_], (tendencies_list[3].dAndt_Ds-tendencies_list[0].dAndt_Ds)[filter_], c ='blue')
axs[0,3].plot(r0.out.t[filter_], (tendencies_list[3].dAndt_T-tendencies_list[0].dAndt_T)[filter_], c ='red')
axs[0,3].plot(r0.out.t[filter_], (tendencies_list[3].dAndt_Cs-tendencies_list[0].dAndt_Cs)[filter_], c ='green')
y = np.arange(-6e-5,7e-5,1e-5)
axs[0,3].fill_betweenx(y,11 + 15/60,15+35/60, alpha = 0.3)
axs[0,3].set_xlabel('Time [UTC]', fontsize = fs)
axs[0,3].set_ylabel('$\Delta$ Tendencies A$_n$ [mg$_{CO_2}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[0,3].set_xlim(4,20)
axs[0,3].set_xticks(x_axis_ticks)
axs[0,3].set_ylim(-6e-5,6e-5)
axs[0,3].grid()
axs[0,3].annotate('TEM-ADV' , xy = (8,5*1e-5,), ha='center', fontsize = fs)

# Leaf transpiration
axs[1,0].set_title('(9e)', fontsize = fs+2)
axs[1,0].plot(r0.out.t[filter_], r0.out.LEleaf[filter_]/Lv*1e6, label = 'Control', c ='k')
axs[1,0].plot(r1.out.t[filter_], r1.out.LEleaf[filter_]/Lv*1e6, label = 'PAR-CLD', c ='k', linestyle ='--')
axs[1,0].plot(r1.out.t[filter_], r2.out.LEleaf[filter_]/Lv*1e6, label = 'VPD-ENT', c ='k', linestyle =':')
axs[1,0].plot(r1.out.t[filter_], r3.out.LEleaf[filter_]/Lv*1e6, label = 'TEM-ADV', c ='k', linestyle ='-.')
axs[1,0].set_xlabel('Time [UTC]', fontsize = fs)
axs[1,0].set_ylabel('TR$_{leaf}$ [mg$_{H_2O}$ m$^{-2}$ s$^{-1}$]', fontsize = fs)
axs[1,0].set_xlim(4,20)
axs[1,0].set_xticks(x_axis_ticks)
axs[1,0].grid()

# Difference of tendencies
axs[1,1].set_title('(9f)', fontsize = fs+2)
axs[1,1].plot(r0.out.t[filter_], (tendencies_list[1].dTRdt-tendencies_list[0].dTRdt)[filter_]*1e6, c ='k')
axs[1,1].plot(r0.out.t[filter_], (tendencies_list[1].dTRdt_PAR-tendencies_list[0].dTRdt_PAR)[filter_]*1e6, c ='orange')
axs[1,1].plot(r0.out.t[filter_], (tendencies_list[1].dTRdt_Ds-tendencies_list[0].dTRdt_Ds)[filter_]*1e6, c ='blue')
axs[1,1].plot(r0.out.t[filter_], (tendencies_list[1].dTRdt_T-tendencies_list[0].dTRdt_T)[filter_]*1e6, c ='red')
axs[1,1].plot(r0.out.t[filter_], (tendencies_list[1].dTRdt_Cs-tendencies_list[0].dTRdt_Cs)[filter_]*1e6, c ='green')
y = np.arange(-0.0135,0.0235,0.01)
axs[1,1].fill_betweenx(y,10,14, alpha = 0.3)
axs[1,1].set_xlabel('Time [UTC]', fontsize = fs)
axs[1,1].set_ylabel('$\Delta$ Tendencies TR$_{leaf}$ [mg$_{H_2O}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[1,1].set_xlim(4,20)
axs[1,1].set_xticks(x_axis_ticks)
axs[1,1].set_ylim(-0.0135,0.0135)
axs[1,1].grid()
axs[1,1].annotate('PAR-CLD' , xy = (8,0.011), ha='center', fontsize = fs)

axs[1,2].set_title('(9g)', fontsize = fs+2)
axs[1,2].plot(r0.out.t[filter_], (tendencies_list[2].dTRdt-tendencies_list[0].dTRdt)[filter_]*1e6, c ='k')
axs[1,2].plot(r0.out.t[filter_], (tendencies_list[2].dTRdt_PAR-tendencies_list[0].dTRdt_PAR)[filter_]*1e6, c ='orange')
axs[1,2].plot(r0.out.t[filter_], (tendencies_list[2].dTRdt_Ds-tendencies_list[0].dTRdt_Ds)[filter_]*1e6, c ='blue')
axs[1,2].plot(r0.out.t[filter_], (tendencies_list[2].dTRdt_T-tendencies_list[0].dTRdt_T)[filter_]*1e6, c ='red')
axs[1,2].plot(r0.out.t[filter_], (tendencies_list[2].dTRdt_Cs-tendencies_list[0].dTRdt_Cs)[filter_]*1e6, c ='green')
y = np.arange(-0.0135/2,0.0235/2, 0.01/2)
axs[1,2].fill_betweenx(y,6 + 20/60,15+35/60, alpha = 0.3)
axs[1,2].set_xlabel('Time [UTC]', fontsize = fs)
axs[1,2].set_ylabel('$\Delta$ Tendencies TR$_{leaf}$ [mg$_{H_2O}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[1,2].set_xlim(4,20)
axs[1,2].set_xticks(x_axis_ticks)
axs[1,2].set_ylim(-0.0135/2,0.0135/2)
axs[1,2].grid()
axs[1,2].annotate('VPD-ENT' , xy = (8,0.011/2), ha='center', fontsize = fs)

axs[1,3].set_title('(9h)', fontsize = fs+2)
axs[1,3].plot(r0.out.t[filter_], (tendencies_list[3].dTRdt-tendencies_list[0].dTRdt)[filter_]*1e6, c ='k')
axs[1,3].plot(r0.out.t[filter_], (tendencies_list[3].dTRdt_PAR-tendencies_list[0].dTRdt_PAR)[filter_]*1e6, c ='orange')
axs[1,3].plot(r0.out.t[filter_], (tendencies_list[3].dTRdt_Ds-tendencies_list[0].dTRdt_Ds)[filter_]*1e6, c ='blue')
axs[1,3].plot(r0.out.t[filter_], (tendencies_list[3].dTRdt_T-tendencies_list[0].dTRdt_T)[filter_]*1e6, c ='red')
axs[1,3].plot(r0.out.t[filter_], (tendencies_list[3].dTRdt_Cs-tendencies_list[0].dTRdt_Cs)[filter_]*1e6, c ='green')
y = np.arange(-0.0135,0.0235, 0.01)
axs[1,3].fill_betweenx(y,11 + 15/60,15+35/60, alpha = 0.3)
axs[1,3].set_xlabel('Time [UTC]', fontsize = fs)
axs[1,3].set_ylabel('$\Delta$ Tendencies TR$_{leaf}$ [mg$_{H_2O}$ m$^{-2}$ s$^{-2}$]', fontsize = fs)
axs[1,3].set_xlim(4,20)
axs[1,3].set_xticks(x_axis_ticks)
axs[1,3].set_ylim(-0.0135,0.0135)
axs[1,3].grid()
axs[1,3].annotate('TEM-ADV' , xy = (8,0.011), ha='center', fontsize = fs)

axs[0,0].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[0,1].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[0,2].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[0,3].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[1,0].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[1,1].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[1,2].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
axs[1,3].axvline(12+2/60+27/3600, linewidth = 1., zorder = 2, color = 'orange', linestyle = '--')
fig.tight_layout()

# Create legend
figLegend1 = plt.figure(figsize = (7,1), dpi = 350)
figLegend1.legend([line_An_CONTROL, line_An_PAR_CLD, line_An_VPD_ENT, line_An_TEM_ADV], \
                  ["Control", "PAR-CLD","VPD-ENT","TEM-ADV"], ncol = 2, loc = "center")
figLegend1.savefig('Fig9_scenarios_legend.png')
figLegend2 = plt.figure(figsize = (7,1), dpi = 350)
figLegend2.legend([line_tend_net, line_tend_PAR, line_tend_VPD, line_tend_TEM,\
                  line_tend_Cs], [r"$\Delta$ Total term", r"$\Delta$ PAR term",\
                                  r"$\Delta$ VPD term", r"$\Delta$ T term", \
                                  r"$\Delta$ C$_a$ term"], ncol = 5, loc = "center")
figLegend2.savefig('Fig9_tendencies_scenarios_legend.png')
