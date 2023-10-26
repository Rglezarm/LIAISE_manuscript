#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:52:45 2023

@author: raquelgonzalezarmas
"""

"""
===> Function that calculates the tendencies_all from CLASS with A-gs run
"""


import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/CLASS_Raquel/modelpy/')
sys.path.append('/Users/raquelgonzalezarmas/Desktop/PhD_Wageningen/1er-paper/models/')
sys.path.append('/Users/raquelgonzalezarmas/Desktop/PhD_Wageningen/Code/Numerical-Methods/')
from Numerical_Differentiation import get_secondOrder_derivative
sys.path.append('/Users/raquelgonzalezarmas/Desktop/PhD_Wageningen/1er-paper/models/Control_simulation/')


# Needed constant and functions
def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

def qsat(T,p):
    return 0.622 * esat(T) / p

# Common needed parameters
mu = 1.6                             # 
rho = 1.2                            # density of air [kg m-3]
    
def calculate_tendencies(r1):
    # Create an empty array with the columns needed.
    tendencies_all = np.zeros((len(r1.out.t),36))
    col_names = ["dPARdt", "dDsdt", "dTdt", "dedt", "dCsdt", "dw2dt",\
                 "dgswdt", "dgswdt_sum_process", "dgswdt_sum_model",\
                 "dgswdPAR", "dgswdDs_T", "dgswdT_Ds", "dgswdCs", "dgswdw2",\
                 "dgswdT", "dgswde",\
                 "dTRdt", "dTRdt_sum_process", "dTRdt_sum_model",\
                 "dTRdPAR", "dTRdDs_T", "dTRdT_Ds", "dTRdCs", "dTRdw2",\
                 "dTRdT", "dTRde",\
                 "dAndt", "dAndt_sum_process", "dAndt_sum_model",\
                 "dAndPAR", "dAndDs_T", "dAndT_Ds", "dAndCs", "dAndw2",\
                 "dAndT", "dAnde"]
    tendencies_process = np.zeros((len(r1.out.t),18))
    col_names_process = ["dgswdt", "dgswdt_PAR", "dgswdt_Ds", "dgswdt_T",\
                         "dgswdt_Cs", "dgswdt_w2",\
                         "dAndt", "dAndt_PAR", "dAndt_Ds", "dAndt_T",\
                         "dAndt_Cs", "dAndt_w2",\
                         "dTRdt", "dTRdt_PAR", "dTRdt_Ds", "dTRdt_T",\
                         "dTRdt_Cs", "dTRdt_w2"]
    # Calculate temporal derivatives of environemntal variables
    t   = r1.out.t                                               # [h]
    PAR = r1.out.PAR                                             # [W/m2]
    Ds  = r1.out.Ds                                              # [Pa]
    T   = r1.out.T0_105m                                         # [K]
    e = r1.out.e0_105m
    co2abs = r1.out.co2abs                                       # [mgCO2/(m3*s)]
    w2  = r1.out.w2                                              # [m3/m3]
    
    t_seconds = t*3600
    dPARdt = get_secondOrder_derivative(t_seconds, PAR)            # [W/(m2*s)]
    dDsdt  = get_secondOrder_derivative(t_seconds, Ds)             # [Pa/s]
    dedt  = get_secondOrder_derivative(t_seconds, e)               # [Pa/s]
    dTdt  = get_secondOrder_derivative(t_seconds, T)               # [K/s]
    dCsdt  = get_secondOrder_derivative(t_seconds, co2abs)         # [mgCO2/(m3*s)]
    dw2dt = get_secondOrder_derivative(t_seconds, w2)              # [m3/m3/s]
    
    tendencies_all[:,0] = dPARdt
    tendencies_all[:,1] = dDsdt
    tendencies_all[:,2] = dTdt
    tendencies_all[:,3] = dedt
    tendencies_all[:,4] = dCsdt
    tendencies_all[:,5] = dw2dt
    
    # Saving and transforming needed variables and parameters
    ## needed variables
    alphac = r1.out.alphac               # Light use efficiency (mg/J)
    Am = r1.out.Am                       # CO2 primary productivity (mg/m2/s)
    Ammax = r1.out.Ammax                 # Maximal CO2 primary productivity (mg/m2/s)
    Rd = r1.out.Rdark                    # CO2 dark respiration (mg/m2/s)
    Dstar = r1.out.Dstar                 # Standarized vapor pressure deficit (Pa). Not sure of the physical meaning
    D0 = r1.out.D0                       # Leaf to air vapor pressure deficit when stomata close (Pa)
    fmin = r1.out.fmin                   # Minimum value of Cfrac (-)
    fmin0 = r1.out.fmin0                 # Interemediate conductance to calculate fmin (m/s)
    gm = r1.out.gm                       # Mesophyll conductance (m/s)
    cfrac = r1.out.cfrac                 # Fraction of the concentrations (ci-gamma)/(co2abs-gamma) (-)
    ci = r1.out.ci                       # Internal CO2 concentration (mg CO2/(m3))
    co2comp = r1.out.CO2comp             # CO2 compensation point (mg/m3)
    
    ## needed parameters
    f0 = r1.f0[0]                        # Maximal value of Cfrac (-)
    a1 = 1/(1-f0)                        # Parameter to calculate glc (-)
    ad = r1.ad[0]/1000                   # Regression coefficient to calculate Cfrac (Pa-1)
    gmin = r1.gmin[0]                    # cuticular (minimum) conductance to water [m s-1]
    gminc = r1.gmin[0]/1.6               # cuticular (minimum) conductance to carbon dioxide [m s-1]
    Q10co2 = r1.Q10CO2[0]                # function parameter to calculate CO2 compensation concentration [-]
    Q10gm = r1.Q10gm[0]                  # function parameter to calculate mesophyll conductance [-]
    Q10Am = r1.Q10Am[0]                  # function parameter to calculate maximum CO2 primary productivity [-]
    T1gm =  r1.T1gm[0]                   # lower reference temperature to calculate mesophyll conductance gm [K]                                                                           
    T2gm = r1.T2gm[0]                    # higher reference temperature to calculate mesophyll conductance gm [K]
    T1Am = r1.T1Am[0]                    # lower reference temperature to calculate maximum CO2 primary productivity [K]
    T2Am = r1.T2Am[0]                    # higher reference temperature to calculate maximum CO2 primary productivity [K]
    alpha0 = r1.alpha0[0]                # Light use efficiency at low light (mg CO2/J)
    Kx    = r1.Kx[0]                     # extinction coefficient PAR (-)
    fstr = r1.out.fstr                   # Soil water stress function (-)      
    
    # Calculate total tendencies_all
    Am = r1.out.Am           # Primary production per area of leaf      
    Agstar = r1.out.Agstar   # Gross primary production per area of leaf without any soil water stress
    Ag = r1.out.Ag           # Gross primary production per area of leaf      
    Aln = r1.out.Aln         # Net assimilation rate per area of leaf
    glc = r1.out.glc         # Stomatal conductance to carbon dioxide [m/s]
    glw = mu*glc             # Stomatal conductance to carbon dioxide [m/s]
    
    dAmdt  = get_secondOrder_derivative(t_seconds, Am)             # CO2 primary productivity at leaf level (mg/m2leaf/s)  
    dAgstardt  = get_secondOrder_derivative(t_seconds, Agstar)     # CO2 gross primary productivity at leaf level (mg/m2leaf/s)
    dAgdt  = get_secondOrder_derivative(t_seconds, Ag)             # CO2 gross primary productivity at leaf level (mg/m2leaf/s)
    dAnldt = get_secondOrder_derivative(t_seconds, Aln)            # net flux of CO2 into the plant (mg/m2/s) 
    dglcdt = get_secondOrder_derivative(t_seconds, glc)            # [m/(s*2)]
    dglwdt = mu*dglcdt                                     # [m/(s*2)]
    # Calculus of tendencies_all of Am
    ## Cs
    dAmdCs = gm*cfrac*(1-Am/Ammax)
    ## Ds at constant T
    dAmdci = gm*(1-Am/Ammax)
    dcidcfrac = (co2abs-co2comp)
    dcfracdDs_T = -ad 
    dAmdDs_T = dAmdci*dcidcfrac*dcfracdDs_T
    ## T at constant Ds
    dco2compdT  = 0.1*co2comp*np.log(Q10co2)
    dAmmaxdT    = 0.1*Ammax*(np.log(Q10Am)+3*(np.exp(0.3*(T1Am-T))\
            -np.exp(0.3*(T-T2Am)))/(1+np.exp(0.3*(T1Am-T)))/(1+np.exp(0.3*(T-T2Am))))
    dgmdT       = 0.1*gm*(np.log(Q10gm)+3*(np.exp(0.3*(T1gm-T))\
            -np.exp(0.3*(T-T2gm)))/(1+np.exp(0.3*(T1gm-T)))/(1+np.exp(0.3*(T-T2gm))))
    dAmdAmmax   = Am/Ammax-gm*cfrac*(co2abs-co2comp)/Ammax*(1-Am/Ammax)
    dAmdgm      = cfrac*(co2abs-co2comp)*(1-Am/Ammax)
    dAmdco2comp = -gm*cfrac*(1-Am/Ammax)
    dAmdci      = gm*(1-Am/Ammax)
    dcidco2comp = 1-cfrac
    dcidcfrac   = co2abs-co2comp
    dcfracdfmin = Ds/D0
    dcfracdD0   = (f0-fmin)*Ds/D0**2
    dD0dfmin    = -1/ad
    dfmindgm    = gmin/1.6/gm/np.sqrt(fmin0**2+4/1.6*gmin*gm)-fmin/gm
    dfmindfmin0 = -fmin/(2*gm*fmin+fmin0)
    dfmin0dgm   = -1/9
    dAmdT_Ds    = dAmdAmmax*dAmmaxdT + dAmdgm*dgmdT + dAmdco2comp*dco2compdT\
            + dAmdci*(dcidco2comp*dco2compdT+dcidcfrac*(dcfracdfmin+dcfracdD0*dD0dfmin)\
            *(dfmindfmin0*dfmin0dgm+dfmindgm)*dgmdT)
    ## e
    dAmde = -dAmdDs_T
    # T
    desatdT = 4098.02862*esat(T)/(T-35.86)**2
    dAmdT   = dAmdT_Ds + dAmdDs_T*desatdT

    # Total tendencies_all
    dAmdt_process = dAmdCs*dCsdt + dAmdDs_T*dDsdt + dAmdT_Ds*dTdt
    dAmdt_model   = dAmdCs*dCsdt + dAmde*dedt + dAmdT*dTdt
    
    # Calculus of tendencies_all of Ag*
    ## PAR
    dAgstardPAR = alphac*(1-Agstar/(Am+Rd))
    ## Cs
    dRddAm = 1/9
    dAgstardRd = Agstar/(Am+Rd)-alphac*PAR/(Am+Rd)*(1-Agstar/(Am+Rd))
    dAgstardAm = dAgstardRd
    dAgstardalpha = PAR*(1-Agstar/(Am+Rd))
    dalphadCs = 3 * alpha0 * co2comp / (co2abs + 2*co2comp)**2
    dAgstardCs = (dAgstardRd*dRddAm + dAgstardAm)*dAmdCs + dAgstardalpha*dalphadCs
    ## Ds at constant T
    dAgstardDs_T = (dAgstardRd*dRddAm+dAgstardAm)*dAmdDs_T
    ## T at constant Ds
    dalphadco2comp = - 3*alpha0*co2abs/(co2abs+2*co2comp)**2
    dAgstardT_Ds = (dAgstardRd*dRddAm+dAgstardAm)*dAmdT_Ds + dAgstardalpha*dalphadco2comp*dco2compdT
    
    ## e
    dAgstarde = -dAgstardDs_T
    ## T
    dAgstardT = dAgstardT_Ds + dAgstardDs_T*desatdT

    ## Total tendencies_all
    dAgstardt_process = dAgstardPAR*dPARdt + dAgstardCs*dCsdt + dAgstardDs_T*dDsdt + dAgstardT_Ds*dTdt
    dAgstardt_model   = dAgstardPAR*dPARdt + dAgstardCs*dCsdt + dAgstarde*dedt + dAgstardT*dTdt


    # Calculus of tendencies_all of Ag
    ## PAR
    dAgdPAR = dAgstardPAR*fstr
    ## Cs
    dAgdCs = dAgstardCs*fstr
    ## Ds at constant T
    dAgdDs_T = dAgstardDs_T*fstr
    ## T at constant Ds
    dAgdT_Ds = dAgstardT_Ds*fstr
    ## w2
    SMI = (w2-r1.input.wwilt)/(r1.input.wfc-r1.input.wwilt)   # Soil moisture index
    PCbeta = r1.out.PCbeta
    dbetadw2 = 1/(r1.input.wfc-r1.input.wwilt*PCbeta*np.exp(-PCbeta*SMI))/(1-np.exp(-PCbeta))
    dAgdw2 = Agstar*dbetadw2
    ## e
    dAgde = -dAgdDs_T
    ## T
    dAgdT = dAgdT_Ds + dAgdDs_T*desatdT
    
    ## Total tendencies_all
    dAgdt_process = dAgdPAR*dPARdt + dAgdCs*dCsdt + dAgdDs_T*dDsdt + dAgdT_Ds*dTdt + dAgdw2*dw2dt
    dAgdt_model   = dAgdPAR*dPARdt + dAgdCs*dCsdt + dAgde*dedt + dAgdT*dTdt + dAgdw2*dw2dt
    
    # Calculus of tendencies_all of Anl
    # PAR
    dAnldPAR = dAgdPAR
    # Cs
    dAnldCs = dAgdCs - dRddAm*dAmdCs
    # Ds at constant T
    dAnldDs_T = dAgdDs_T - dRddAm*dAmdDs_T
    # T at constant Ds
    dAnldT_Ds = dAgdT_Ds - dRddAm*dAmdT_Ds
    # w2
    dAnldw2 = dAgdw2
    # e
    dAnlde = dAgde - dRddAm*dAmde
    # T
    dAnldT = dAgdT - dRddAm*dAmdT
    
    # Total tendencies_all
    dAnldt_process = dAnldPAR*dPARdt + dAnldCs*dCsdt + dAnldDs_T*dDsdt + dAnldT_Ds*dTdt + dAnldw2*dw2dt
    dAnldt_model   = dAnldPAR*dPARdt + dAnldCs*dCsdt + dAnlde*dedt + dAnldT*dTdt + dAnldw2*dw2dt

    ## Saving An tendencies_all
    tendencies_all[:,26] = dAnldt
    tendencies_all[:,27] = dAnldt_process
    tendencies_all[:,28] = dAnldt_model
    tendencies_all[:,29] = dAnldPAR
    tendencies_all[:,30] = dAnldDs_T
    tendencies_all[:,31] = dAnldT_Ds
    tendencies_all[:,32] = dAnldCs
    tendencies_all[:,33] = dAnldw2
    tendencies_all[:,34] = dAnlde
    tendencies_all[:,35] = dAnldT
    # Calculus of tendencies_all of glc and glw
    ## PAR
    dglcdAnl = a1/(co2abs-co2comp)/(1+Ds/Dstar)
    dglcdPAR = dglcdAnl*dAnldPAR
    dglwdPAR = mu*dglcdPAR
    ## Cs
    dglcdCs_Anl = -(glc-gminc)/(co2abs-co2comp)
    dglcdCs = dglcdCs_Anl + dglcdAnl*dAnldCs
    dglwdCs = mu*dglcdCs
    ## Ds at constant T
    dglcdDs_AnlT = -(glc-gminc)/(Dstar+Ds)
    dglcdDs_T = dglcdAnl * dAnldDs_T+dglcdDs_AnlT
    dglwdDs_T = mu*dglcdDs_T
    ## T at constant Ds
    dglcdco2comp_AgT = (glc-gminc)/(co2abs-co2comp)
    dglcdT_Ds = dglcdAnl*dAnldT_Ds + dglcdco2comp_AgT*dco2compdT
    dglwdT_Ds = mu*dglcdT_Ds
    ## w2
    dglcdw2 = dglcdAnl*dAnldw2
    dglwdw2 = mu*dglcdw2
    ## e
    dglcde = - dglcdDs_T
    dglwde = mu*dglcde
    ## T
    dglcdT = dglcdT_Ds + dglcdDs_T*desatdT
    dglwdT = mu*dglcdT
    
    ## Total tendencies_all
    dglcdt_process = dglcdPAR*dPARdt + dglcdCs*dCsdt + dglcdDs_T*dDsdt + dglcdT_Ds*dTdt + dglcdw2*dw2dt
    dglcdt_model   = dglcdPAR*dPARdt + dglcdCs*dCsdt + dglcde*dedt + dglcdT*dTdt + dglcdw2*dw2dt
    dglwdt_process = dglwdPAR*dPARdt + dglwdCs*dCsdt + dglwdDs_T*dDsdt + dglwdT_Ds*dTdt + dglwdw2*dw2dt
    dglwdt_model   = dglwdPAR*dPARdt + dglwdCs*dCsdt + dglwde*dedt + dglwdT*dTdt + dglwdw2*dw2dt
    
    ## Save glw tendencies_all
    tendencies_all[:,6] = dglwdt
    tendencies_all[:,7] = dglwdt_process
    tendencies_all[:,8] = dglwdt_model
    tendencies_all[:,9] = dglwdPAR
    tendencies_all[:,10] = dglwdDs_T
    tendencies_all[:,11] = dglwdT_Ds
    tendencies_all[:,12] = dglwdCs
    tendencies_all[:,13] = dglwdw2
    tendencies_all[:,14] = dglwde
    tendencies_all[:,15] = dglwdT
    
    # Calculus of tendencies_all of TRleaf
    wair = rho * e * 0.622/r1.input.Ps
    wi   = rho*qsat(T, r1.input.Ps)
    
    TR = glc*mu*(wi-wair)
    # PAR
    dTRdglw = (wi-wair)           # [kg water/kg air]
    dTRdPAR = dTRdglw*dglwdPAR    # [kg water m-2 s-2]
    
    # Cs
    dTRdCs = dTRdglw*dglwdCs
    
    # w2
    dTRdw2 = dTRdglw*dglwdw2
    
    # Ds at constant T
    dTRdDs_T = dTRdglw*dglwdDs_T+glc*mu*rho*0.622/r1.input.Ps
    
    # T at constant Ds
    dTRdT_Ds = dTRdglw*dglwdT_Ds
    
    dTRdt = get_secondOrder_derivative(t_seconds, TR)             
    dTRdt_process = dTRdPAR*dPARdt + dTRdCs*dCsdt + dTRdDs_T*dDsdt + dTRdT_Ds*dTdt 
    
    
    ## Save TR tendencies_all
    tendencies_all[:,16] = dTRdt
    tendencies_all[:,17] = dTRdt_process
#    tendencies_all[:,18] = dTRdt_model
    tendencies_all[:,19] = dTRdPAR
    tendencies_all[:,20] = dTRdDs_T
    tendencies_all[:,21] = dTRdT_Ds
    tendencies_all[:,22] = dTRdCs
    tendencies_all[:,23] = dTRdw2
#    tendencies_all[:,24] = dTRde
#    tendencies_all[:,25] = dTRdT
    
    ## Save tendencies_process
    tendencies_process[:,0] = dglwdt
    tendencies_process[:,1] = dglwdPAR*dPARdt
    tendencies_process[:,2] = dglwdDs_T*dDsdt
    tendencies_process[:,3] = dglwdT_Ds*dTdt
    tendencies_process[:,4] = dglwdCs*dCsdt
    tendencies_process[:,5] = dglwdw2*dw2dt
    tendencies_process[:,6] = dAnldt
    tendencies_process[:,7] = dAnldPAR*dPARdt
    tendencies_process[:,8] = dAnldDs_T*dDsdt
    tendencies_process[:,9] = dAnldT_Ds*dTdt
    tendencies_process[:,10] = dAnldCs*dCsdt
    tendencies_process[:,11] = dAnldw2*dw2dt
    tendencies_process[:,12] = dTRdt
    tendencies_process[:,13] = dTRdPAR*dPARdt
    tendencies_process[:,14] = dTRdDs_T*dDsdt
    tendencies_process[:,15] = dTRdT_Ds*dTdt
    tendencies_process[:,16] = dTRdCs*dCsdt
    tendencies_process[:,17] = dTRdw2*dw2dt
    
    # Save tendencies_all as a pd.Dataframe
    tendencies_all_documented = pd.DataFrame(data = tendencies_all, columns = col_names)
    
    # Save tendencies_process as a pd.Dataframe
    tendencies_process_documented = pd.DataFrame(data = tendencies_process, columns = col_names_process)
    
    # Return tendencies_all calculated
    return tendencies_all_documented, tendencies_process_documented


def reconstruct_leafgasexchange(r1):
    
    tendencies_all_documented, tendencies_process = calculate_tendencies(r1)
    
    reconstruction = np.zeros((len(r1.out.t), 15))
    
    col_names = ["gsw_PAR", "gsw_Ds", "gsw_T", "gsw_Cs", "gsw_w2",\
                "An_PAR", "An_Ds", "An_T", "An_Cs", "An_w2",\
                "TR_PAR", "TR_Ds", "TR_T", "TR_Cs", "TR_w2"]
    
    # Calculate total tendencies_all
    gsw = mu*r1.out.glc         # Stomatal conductance to carbon dioxide
    An = r1.out.Aln         # Net assimilation rate per area of leaf
    wair = rho*r1.out.e0_105m * 0.622/r1.input.Ps
    wi   = rho*qsat(r1.out.T0_0105m, r1.input.Ps)
    TR = gsw*(wi-wair)
    
    gsw_PAR = np.zeros(len(r1.out.t))
    gsw_Ds = np.zeros(len(r1.out.t))
    gsw_T = np.zeros(len(r1.out.t))
    gsw_Cs = np.zeros(len(r1.out.t))
    gsw_w2 = np.zeros(len(r1.out.t))

    
    An_PAR = np.zeros(len(r1.out.t))
    An_Ds = np.zeros(len(r1.out.t))
    An_T = np.zeros(len(r1.out.t))
    An_Cs = np.zeros(len(r1.out.t))
    An_w2 = np.zeros(len(r1.out.t))
    
    TR_PAR = np.zeros(len(r1.out.t))
    TR_Ds = np.zeros(len(r1.out.t))
    TR_T = np.zeros(len(r1.out.t))
    TR_Cs = np.zeros(len(r1.out.t))
    TR_w2 = np.zeros(len(r1.out.t))
    
    # Temporal step of the model
    dt = r1.input.dt
    
    for i in np.arange(len(r1.out.t)-1): 
        gsw_PAR[i+1] = gsw_PAR[i]+ tendencies_process.dgswdt_PAR[i+1]*dt
        gsw_Ds[i+1]= gsw_Ds[i]+ tendencies_process.dgswdt_Ds[i+1]*dt
        gsw_T[i+1]= gsw_T[i]+ tendencies_process.dgswdt_T[i+1]*dt
        gsw_Cs[i+1]= gsw_Cs[i]+ tendencies_process.dgswdt_Cs[i+1]*dt
        gsw_w2[i+1]= gsw_w2[i]+ tendencies_process.dgswdt_w2[i+1]*dt
        
        An_PAR[i+1]=An_PAR[i]+ tendencies_process.dAndt_PAR[i+1]*dt
        An_Ds[i+1]=An_Ds[i]+ tendencies_process.dAndt_Ds[i+1]*dt
        An_T[i+1]=An_T[i]+ tendencies_process.dAndt_T[i+1]*dt
        An_Cs[i+1]=An_Cs[i]+ tendencies_process.dAndt_Cs[i+1]*dt
        An_w2[i+1]=An_w2[i]+ tendencies_process.dAndt_w2[i+1]*dt
        
        TR_PAR[i+1]=TR_PAR[i]+ tendencies_process.dTRdt_PAR[i+1]*dt
        TR_Ds[i+1]=TR_Ds[i]+ tendencies_process.dTRdt_Ds[i+1]*dt
        TR_T[i+1]=TR_T[i]+ tendencies_process.dTRdt_T[i+1]*dt
        TR_Cs[i+1]=TR_Cs[i]+ tendencies_process.dTRdt_Cs[i+1]*dt
        TR_w2[i+1]=TR_w2[i]+ tendencies_process.dTRdt_w2[i+1]*dt
        
    reconstruction[:,0] = gsw_PAR
    reconstruction[:,1] = gsw_Ds
    reconstruction[:,2] = gsw_T
    reconstruction[:,3] = gsw_Cs
    reconstruction[:,4] = gsw_w2
    
    reconstruction[:,5] = An_PAR
    reconstruction[:,6] = An_Ds
    reconstruction[:,7] = An_T
    reconstruction[:,8] = An_Cs
    reconstruction[:,9] = An_w2
    
    reconstruction[:,10] = TR_PAR
    reconstruction[:,11] = TR_Ds
    reconstruction[:,12] = TR_T
    reconstruction[:,13] = TR_Cs
    reconstruction[:,14] = TR_w2
    
    print(reconstruction)
    reconstruction_documented = pd.DataFrame(data = reconstruction, columns = col_names)
    
    return reconstruction_documented
    
    
    
    
