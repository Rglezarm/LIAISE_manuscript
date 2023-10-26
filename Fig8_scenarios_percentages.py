#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:03:00 2023

@author: raquelgonzalezarmas
"""

"""
    ===> Script that creates Fig. 8 of LIAISE paper: it calculates relative 
    percentages between the three perturbed experiments and the control case
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/PhD_Wageningen/1er-paper/models/Control_simulation/')
import sensitivity_control_cases as final_cases
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-Wage'+\
                'ningenUniversity&Research/PhD_Wageningen/Code/Numerical-'+\
                    'Methods/Integration.py')
import Integration as integration
#%%
r1    = final_cases.control()
r2    = final_cases.control_cloud()
r3    = final_cases.control_ent_DA()
r4    = final_cases.control_adv_CA()

t = r1.out.t

#%%
# plt.figure()
# plt.plot(r1.out.t[filter_], r1.out.ra[filter_], label = 'Control')
# plt.plot(r3.out.t[filter_], r3.out.ra[filter_], label = 'Ent. D.A.')
# plt.legend()
#%%
Lv         = 2.5e6                 # heat of vaporization [J kg-1]

def percentages_runs(run_control, run_cloud, t_initial, t_final):
    filter_ = (t>t_initial)&(t<=t_final)
    
    print("Percentages of gs")
    gs_control = 1/run_control.out.rs  # m/s
    gs_cloud = 1/run_cloud.out.rs      # m/s
    exchange_gs_control = integration.area_trapezoidal(gs_control[filter_], 5)
    exchange_gs_cloud = integration.area_trapezoidal(gs_cloud[filter_], 5)
    sum_gs_CONTROL = np.sum(gs_control[filter_])
    sum_gs_CLOUD = np.sum(gs_cloud[filter_])
    P_gs2 = (sum_gs_CLOUD-sum_gs_CONTROL)/sum_gs_CONTROL
    P_gs = (exchange_gs_cloud-exchange_gs_control)/exchange_gs_control
    print(r"gs accumulated control = %.2f  g CO2 m-2"%(exchange_gs_control*1e-3))
    print(r"gs accumulated cloud = %.2f  g CO2 m-2"%(exchange_gs_cloud*1e-3))
    print("Percentage of change of gs (1st method): %.2f"%(P_gs*100))
    print("Percentage of change of gs (2nd method): %.2f"%(P_gs2*100))
    print("Percentage in change of gs (3rd method): %.2f"%((np.mean(gs_cloud[filter_])-np.mean(gs_control[filter_]))/np.mean(gs_control[filter_])*100))
    
    
    
    print("Percentages of Anc")
    Anc_control = -run_control.out.wCO2A  # mg CO2/m2/s
    Anc_cloud = -run_cloud.out.wCO2A      # mg CO2/m2/s
    exchange_Anc_control = integration.area_trapezoidal(Anc_control[filter_], 5)
    exchange_Anc_cloud = integration.area_trapezoidal(Anc_cloud[filter_], 5)
    sum_Anc_CONTROL = np.sum(Anc_control[filter_])
    sum_Anc_CLOUD = np.sum(Anc_cloud[filter_])
    P_Anc2 = (sum_Anc_CLOUD-sum_Anc_CONTROL)/sum_Anc_CONTROL
    P_Anc = (exchange_Anc_cloud-exchange_Anc_control)/exchange_Anc_control
    print(r"A$_{nc}$ accumulated control = %.2f  g CO2 m-2"%(exchange_Anc_control*1e-3))
    print(r"A$_{nc}$ accumulated cloud = %.2f  g CO2 m-2"%(exchange_Anc_cloud*1e-3))
    print("Percentage of change of A$_{nc}$ (1st method): %.2f"%(P_Anc*100))
    print("Percentage of change of A$_{nc}$ (2nd method): %.2f"%(P_Anc2*100))
    print("Percentage in change of A$_{nc}$ (3rd method): %.2f"%((np.mean(Anc_cloud[filter_])-np.mean(Anc_control[filter_]))/np.mean(Anc_control[filter_])*100))
    
    
    print("Percentages of NEE")
    NEE_control = -run_control.out.wCO2  # mg CO2/m2/s
    NEE_cloud = -run_cloud.out.wCO2      # mg CO2/m2/s
    exchange_CO2_control = integration.area_trapezoidal(NEE_control[filter_], 5)
    exchange_CO2_cloud = integration.area_trapezoidal(NEE_cloud[filter_], 5)
    sum_NEE_CONTROL = np.sum(NEE_control[filter_])
    sum_NEE_CLOUD = np.sum(NEE_cloud[filter_])
    P_NEE2 = (sum_NEE_CLOUD-sum_NEE_CONTROL)/sum_NEE_CONTROL
    P_NEE = (exchange_CO2_cloud-exchange_CO2_control)/exchange_CO2_control
    print(r"NEE accumulated control = %.2f  g CO2 m-2"%(exchange_CO2_control*1e-3))
    print(r"NEE accumulated cloud = %.2f  g CO2 m-2"%(exchange_CO2_cloud*1e-3))
    print("Percentage of change of NEE (1st method): %.2f"%(P_NEE*100))
    print("Percentage of change of NEE (2nd method): %.2f"%(P_NEE2*100))
    print("Percentage in change of NEE (3rd method): %.2f"%((np.mean(NEE_cloud[filter_])-np.mean(NEE_control[filter_]))/np.mean(NEE_control[filter_])*100))
    
    print("Percentages of ET")
    ET_control = run_control.out.LE/Lv*1e6
    ET_cloud   = run_cloud.out.LE/Lv*1e6
    exchange_H2O_control = integration.area_trapezoidal(ET_control[filter_], 5)
    exchange_H2O_cloud = integration.area_trapezoidal(ET_cloud[filter_], 5)
    sum_LE_CONTROL = np.sum(run_control.out.LE[filter_])
    sum_LE_CLOUD = np.sum(run_cloud.out.LE[filter_])
    P_ET2 = (sum_LE_CLOUD-sum_LE_CONTROL)/sum_LE_CONTROL
    P_ET = (exchange_H2O_cloud-exchange_H2O_control)/exchange_H2O_control
    print(r"ET accumulated control = %.2f  kg water m-2"%(exchange_H2O_control*1e-6))
    print(r"ET accumulated cloud = %.2f  kg water m-2"%(exchange_H2O_cloud*1e-6))
    print("Percentage of change of ET (1st method): %.2f"%(P_ET*100))
    print("Percentage of change of ET (2nd method): %.2f"%(P_ET2*100))
    print("Percentage in change of ET (3rd method): %.2f"%((np.mean(ET_cloud[filter_])-np.mean(ET_control[filter_]))/np.mean(ET_control[filter_])*100))
    
    print("Percentages of gsw")
    gsw_control = run_control.out.glw   # m/s
    gsw_cloud   = run_cloud.out.glw     # m/s
    exchange_air_control = integration.area_trapezoidal(gsw_control[filter_], 5)
    exchange_air_cloud = integration.area_trapezoidal(gsw_cloud[filter_], 5)
    sum_gsw_CONTROL = np.sum(gsw_control[filter_])
    sum_gsw_CLOUD = np.sum(gsw_cloud[filter_])
    P_gsw2 = (sum_gsw_CLOUD-sum_gsw_CONTROL)/sum_gsw_CONTROL
    P_gsw = (exchange_air_cloud-exchange_air_control)/exchange_air_control
    print(r"gsw accumulated control = %.4f  m3 air m-2 leaf"%(exchange_air_control))
    print(r"gsw accumulated cloud = %.4f  m3 air m-2 leaf"%(exchange_air_cloud))
    print("Percentage of change of gsw (1st method): %.2f"%(P_gsw*100))
    print("Percentage of change of gsw (2nd method): %.2f"%(P_gsw2*100))
    print("Percentage in change of gsw (3rd method): %.2f"%((np.mean(gsw_cloud[filter_])-np.mean(gsw_control[filter_]))/np.mean(gsw_control[filter_])*100))
    
    print("Percentages of An")
    An_control = run_control.out.Aln   # mg CO2/ m2 /s
    An_cloud   = run_cloud.out.Aln     # mg CO2/ m2 /s
    exchange_leafCO2_control = integration.area_trapezoidal(An_control[filter_], 5)
    exchange_leafCO2_cloud = integration.area_trapezoidal(An_cloud[filter_], 5)
    sum_An_CONTROL = np.sum(An_control[filter_])
    sum_An_CLOUD = np.sum(An_cloud[filter_])
    P_An2 = (sum_An_CLOUD-sum_An_CONTROL)/sum_An_CONTROL
    P_An = (exchange_leafCO2_cloud-exchange_leafCO2_control)/exchange_leafCO2_control
    print(r"An accumulated control = %.4f  g CO2 m-2 leaf"%(exchange_leafCO2_control*1e-3))
    print(r"An accumulated cloud = %.4f  g CO2 m-2 leaf"%(exchange_leafCO2_cloud*1e-3))
    print("Percentage of change of An (1st method): %.2f"%(P_An*100))
    print("Percentage of change of An (2nd method): %.2f"%(P_An2*100))
    print("Percentage in change of An (3rd method): %.2f"%((np.mean(An_cloud[filter_])-np.mean(An_control[filter_]))/np.mean(An_control[filter_])*100))
    
    print("Percentages of TR")
    TR_control = run_control.out.LEleaf/Lv*1e6   # mg H20/ m2 /s
    TR_cloud   = run_cloud.out.LEleaf/Lv*1e6     # mg H20/ m2 /s
    exchange_leafH2O_control = integration.area_trapezoidal(TR_control[filter_], 5)
    exchange_leafH2O_cloud = integration.area_trapezoidal(TR_cloud[filter_], 5)
    sum_TR_CONTROL = np.sum(TR_control[filter_])
    sum_TR_CLOUD = np.sum(TR_cloud[filter_])
    P_TR2 = (sum_TR_CLOUD-sum_TR_CONTROL)/sum_TR_CONTROL
    P_TR = (exchange_leafH2O_cloud-exchange_leafH2O_control)/exchange_leafH2O_control
    print(r"TR accumulated control = %.4f  kg H2O m-2 leaf"%(exchange_leafH2O_control*1e-6))
    print(r"TR accumulated cloud = %.4f  kg H2O m-2 leaf"%(exchange_leafH2O_cloud*1e-6))
    print("Percentage of change of TR (1st method): %.2f"%(P_TR*100))
    print("Percentage of change of TR (2nd method): %.2f"%(P_TR2*100))
    print("Percentage in change of TR (3rd method): %.2f"%((np.mean(TR_cloud[filter_])-np.mean(TR_control[filter_]))/np.mean(TR_control[filter_])*100))
    
    return P_gs, P_Anc, P_NEE, P_ET, P_gsw, P_An, P_TR




#%% Create barplot 

P_gs_r2, P_Anc_r2, P_NEE_r2, P_ET_r2, P_gws_r2, P_An_r2, P_TR_r2 = percentages_runs(r1,r2, 5, 15.5)
P_gs_r3, P_Anc_r3, P_NEE_r3, P_ET_r3, P_gws_r3, P_An_r3, P_TR_r3 = percentages_runs(r1,r3, 5, 15.5)
P_gs_r4, P_Anc_r4, P_NEE_r4, P_ET_r4, P_gws_r4, P_An_r4, P_TR_r4 = percentages_runs(r1,r4, 5, 15.5)

barheights_r2 = np.array([P_gws_r2, P_An_r2, P_TR_r2, P_gs_r2, P_Anc_r2, P_ET_r2, P_NEE_r2,])*100
barheights_r3 = np.array([P_gws_r3, P_An_r3, P_TR_r3, P_gs_r3, P_Anc_r3, P_ET_r3, P_NEE_r3])*100
barheights_r4 = np.array([P_gws_r4, P_An_r4, P_TR_r4, P_gs_r4, P_Anc_r4, P_ET_r4, P_NEE_r4])*100

bar_heights_gsw = np.array([P_gws_r2, P_gws_r3, P_gws_r4])*100
bar_heights_An = np.array([P_An_r2, P_An_r3, P_An_r4])*100
bar_heights_TR = np.array([P_TR_r2, P_TR_r3, P_TR_r4])*100
bar_heights_gs = np.array([P_gs_r2, P_gs_r3, P_gs_r4])*100
bar_heights_GPP = np.array([P_Anc_r2, P_Anc_r3, P_Anc_r4])*100
bar_heights_ET = np.array([P_ET_r2, P_ET_r3, P_ET_r4])*100
bar_heights_NEE = np.array([P_NEE_r2, P_NEE_r3, P_NEE_r4])*100

br1 = np.array([0,1,2,4,5,6,7])
width = 0.22


fig = plt.figure(dpi = 350, figsize = (14,4))
ax = fig.gca()
fs = 14 


ax.bar(np.array([0,10,20]), bar_heights_gsw, width = 0.9, color = 'C2', edgecolor = 'k', linestyle ='-', label = '$g_{s}$')
ax.bar(np.array([1,11,21]), bar_heights_An, width = 0.9, color = 'C2', edgecolor = 'k', linestyle ='-', label = '$A_{n}$')
ax.bar(np.array([2,12,22]), bar_heights_TR, width = 0.9, color = 'C2', edgecolor = 'k', linestyle ='-', label = '$TR_{leaf}$')
ax.bar(np.array([4,14,24]), bar_heights_gs, width = 0.9, color = 'C0', edgecolor = 'k', linestyle ='-', label = '$g_{surf}$')
ax.bar(np.array([5,15,25]), bar_heights_GPP, width = 0.9, color = 'C0', edgecolor = 'k', linestyle ='-', label = '$A_{nc}$')
ax.bar(np.array([6,16,26]), bar_heights_ET, width = 0.9, color = 'C0', edgecolor = 'k', linestyle ='-', label = '$ET$')
ax.bar(np.array([7,17,27]), bar_heights_NEE, width = 0.9, color = 'C0', edgecolor = 'k', linestyle ='-', label = '$-NEE$')

ax.grid(axis ='y')
ax.set_axisbelow(True)
#ax.legend(loc = 'upper center', ncol =7, fontsize = fs)


ax.set_ylim(-15,15)
ax.set_ylabel(r'Decrease [$\%$]            Increase [$\%$]',  fontsize = fs)
ax.axhline(0, c = 'k', linewidth = 0.8)
ax.axvline(8.5, c = 'k', linestyle = ':', linewidth = 0.8)
ax.axvline(18.5, c = 'k', linestyle = ':', linewidth = 0.8)

x_annotate = np.array([0,1,2,4,5,6,7,10,11,12,14,15,16,17,20,21,22,24,25,26,27])
y_annotate = 8.5*np.ones(len(x_annotate))
labels_annotate = [r'$g_{s}$', r'$A_n$', r'$TR_{leaf}$',r'$g_{surf}$', '$A_{nc}$', 'ET','-NEE',\
                   r'$g_{s}$', r'$A_n$', r'$TR_{leaf}$',r'$g_{surf}$', '$A_{nc}$', 'ET','-NEE',\
                   r'$g_{s}$', r'$A_n$', r'$TR_{leaf}$',r'$g_{surf}$', '$A_{nc}$', 'ET','-NEE']
for i in np.arange(len(x_annotate)):
    ax.annotate(labels_annotate[i], (x_annotate[i], y_annotate[i]), ha='center', fontsize = fs-4)

ax.annotate('Leaf level', (1,12.5), ha = 'center', fontsize = fs-1)
ax.annotate('Canopy level', (5.5,12.5), ha = 'center', fontsize = fs-1)
ax.annotate('Leaf level', (11,12.5), ha = 'center', fontsize = fs-1)
ax.annotate('Canopy level', (15.5,12.5), ha = 'center', fontsize = fs-1)
ax.annotate('Leaf level', (21,12.5), ha = 'center', fontsize = fs-1)
ax.annotate('Canopy level', (25.5,12.5), ha = 'center', fontsize = fs-1)

ax.set_xticks([3.5,14,24])
ax.set_xticklabels(['PAR-CLD','VPD-ENT', 'TEM-ADV'], fontsize = fs)

#%% Barplot function for only 2 runs
def barplot(run_control, run_cloud, t_initial, t_final):
    fig = plt.figure(dpi = 350)
    ax = fig.gca()
    
    P_NEE, P_ET, P_gws, P_An, P_TR = percentages_runs(run_control,run_cloud,t_initial,t_final)
    barheights = np.array([P_NEE, P_ET, P_gws, P_An, P_TR])*100
    br1 = np.arange(len(barheights))
    br1 = ['NEE','ET', r'g$_{s}$', r'A$_n$', 'TR']
    ax.bar(br1, barheights)
    ax.set_ylim(-30,30)
    ax.set_ylabel(r'Decrease [$\%$]            Increase [$\%$]')
    ax.axhline(0, c = 'k', linewidth = 0.8)

