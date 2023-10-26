#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:02:30 2023

@author: raquelgonzalezarmas
"""


"""
    ===> Script that calculates response curves of photosynthesis to PAR and Ci for
    alfalfa LIAISE. This figure is the one used in Appendix A of my paper 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Ags import Ags



"""
(1) Open the data
"""
A_Ci = pd.read_excel('/Users/raquelgonzalezarmas/OneDrive - Wageningen University '+\
                     '& Research/Ecophisiology-LIAISE/A-ci&PAR/A_ci_for_fit.xlsx')
A_PAR = pd.read_excel('/Users/raquelgonzalezarmas/OneDrive - Wageningen University '+\
                      '& Research/Ecophisiology-LIAISE/A-ci&PAR/A_par_for_fit.xlsx')


"""
(2) Conversion factors
"""
#%% Conversion of photon flux density at the photosynthetic active range to photosynthetic active radiation
Na         = 6.0221408e23        # Avogadro's number (# of particles in 1 mol)
h          = 6.62607015e-34      # Planck's constant (J*s)
c          = 299792458           # Light speed constant (m/s)
lambda_PAR = 550e-9              # Wavelenght of mean photon in the active photosynthetic range (m)

from_Q_to_PAR = Na*h*c/lambda_PAR*1e-6

#%% Conversion from ppm to mg CO2/m3

mco2       =  44.                # molecular weight CO2 (g mol -1)
mair       =  28.9               # molecular weight air (g mol -1)
rho        =  1.2                # density of air (kg m-3)

from_ppm_to_mg_m3 = mco2/mair*rho

#%% Conversion from mgCO2/(m2*s) to micromolCO2/(m2*s)
from_mg_to_micromol      = 1/mco2*1e3  
#%%

    
"""
(2) Choice of parameters
"""

Ags_param_Alfalfa = {'CO2comp298': 68.5*rho,  # CO2 compensation concentration (mg m-3)
                    'Q10CO2':1.5,      # function parameter to calculate CO2 compensation concentration (-)
                    'gm298':10.2,       # mesophyill conductance at 298 K (mm s-1)
                    'Ammax298': 3.024,   # CO2 maximal primary productivity (mg m-2 s-1)
                    'Q10gm': 2.0,      # function parameter to calculate mesophyll conductance (-)
                    'T1gm': 278,       # reference temperature to calculate mesophyll conductance gm (K)
                    'T2gm': 306,       # reference temperature to calculate mesophyll conductance gm (K)
                    'Q10Am': 2.0,      # function parameter to calculate maximal primary profuctivity Ammax (-)
                    'T1Am': 281,       # reference temperature to calculate maximal primary profuctivity Ammax (K)
                    'T2Am': 311,       # reference temperature to calculate maximal primary profuctivity Ammax (K)
                    'f0': 0.89,        # maximum value Cfrac (-)
                    'ad': 0.07,        # regression coefficient to calculate Cfrac (kPa-1)
                    'alpha0': 0.0265,   # initial low light conditions (mg J-1)
                    'Kx': 0.7,         # extinction coefficient PAR (-)
                    'gmin':0.25e-3}    # cuticular (minimum) conductance to water (m s-1)




"""
(2) Plot A-Ci curves
"""

fig, axs = plt.subplots(2,dpi = 400, figsize = (8,8))
ax = axs[0]

# Plot measurements, each curve with a distintic color

for curve in np.unique(A_Ci.Curve):
    A_Ci_curve = A_Ci[A_Ci.Curve == curve]
    if curve == 1:
        label = 'OBS'
    else:
        label = None
    ax.scatter(A_Ci_curve.Ci, A_Ci_curve.Photo, label = label, c = 'k', alpha = 0.6)

# Plot models: 
A_modelled_default = Ags(T = A_Ci.Tleaf + 273.15, \
                  PAR = A_Ci.PARi*from_Q_to_PAR, \
                  Cs = 400, \
                  Ci = A_Ci.Ci*from_ppm_to_mg_m3, \
                  parameters = Ags_param_Alfalfa)

ax.scatter(A_Ci.Ci, A_modelled_default*from_mg_to_micromol, label = r'A-g$_s$', color = 'green', alpha = 0.6)

   
ax.set_ylim(0,70)
ax.grid()
ax.set_xlabel(r'C$_i$ [ppm]')
ax.set_ylabel(r'$A_n$ [$\mu mol_{CO2}~m^{-2}~s^{-1}$]')
ax.legend(loc = 'lower center')
new_tick_locations_Ci = np.array([0,200,400,600,800,1000,1200])

def tick_function_Ci(Ci_ppm):
    Ci_mg_m3 = Ci_ppm * from_ppm_to_mg_m3
    return ["%i" % round(z) for z in Ci_mg_m3] 

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(new_tick_locations_Ci)
ax2.set_xticklabels(tick_function_Ci(new_tick_locations_Ci))
ax2.set_xlabel(r'C$_i$ [$mg_{CO2}~m^{-3}$]')

new_tick_locations_An = np.array([0,10,20,30,40,50,60,70])

def tick_function_An(An_mumol_m2_s):
    An_mg_m2_s = An_mumol_m2_s / from_mg_to_micromol
    return ["%.2f" % z for z in An_mg_m2_s] 

ax3 = ax.twinx()
ax3.set_ylim(ax.get_ylim())
ax3.set_yticks(new_tick_locations_An)
ax3.set_yticklabels(tick_function_An(new_tick_locations_An))
ax3.set_ylabel(r'$A_n$ [$mg_{CO2}~m^{-2}~s^{-1}$]')

ax.set_title('(A3a) A$_n$-C$_i$ response curve')


"""
(3) Plot A-PAR curves
"""

ax = axs[1]


# Plot measurements, each curve with a distintic color

for curve in np.unique(A_PAR.Curve):
    if curve == 1:
        label = 'OBS'
    else:
        label = None
    A_PAR_curve = A_PAR[A_PAR.Curve == curve]
    ax.scatter(A_PAR_curve.PARi, A_PAR_curve.Photo, label = label,c = 'k', alpha = 0.6)


# Plot model: 
A_modelled_default = Ags(T = A_PAR.Tleaf + 273.15, \
                  PAR = A_PAR.PARi*from_Q_to_PAR, \
                  Cs = 400, \
                  Ci = A_PAR.Ci*from_ppm_to_mg_m3, \
                   parameters = Ags_param_Alfalfa)

ax.scatter(A_PAR.PARi, A_modelled_default*from_mg_to_micromol, label = r'A-g$_s$', color = 'green', alpha = 0.6)

ax.grid()
ax.legend(loc="lower center")
ax.set_xlabel(r'Q [$\mu mol \gamma~m^{-2}~s^{-1}$]')
ax.set_ylabel(r'$A_n$ [$\mu mol_{CO2}~m^{-2}s^{-1}$]')


new_tick_locations_PAR = np.array([0,200,400,600,800,1000,1200, 1400])

def tick_function_PAR(PAR_micromol):
    PAR_W_m2 = PAR_micromol * from_Q_to_PAR
    return ["%i" % round(z) for z in PAR_W_m2] 

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(new_tick_locations_PAR)
ax2.set_xticklabels(tick_function_PAR(new_tick_locations_PAR))
ax2.set_xlabel(r'PAR [W m$^{-2}$]')

new_tick_locations_An = np.array([-10, 0,10,20,30,40])

def tick_function_An(An_mumol_m2_s):
    An_mg_m2_s = An_mumol_m2_s / from_mg_to_micromol
    return ["%.2f" % z for z in An_mg_m2_s] 

ax3 = ax.twinx()
ax3.set_ylim(ax.get_ylim())
ax3.set_yticks(new_tick_locations_An)
ax3.set_yticklabels(tick_function_An(new_tick_locations_An))
ax3.set_ylabel(r'$A_n$ [$mg_{CO2}~m^{-2}~s^{-1}$]')

ax.set_title('(A3b) A$_n$-PAR response curve')

fig.tight_layout()
#fig.savefig('EL_GRAN_ERROR_DE_RAQUEL_A_PAR_LaCendrosa.png')