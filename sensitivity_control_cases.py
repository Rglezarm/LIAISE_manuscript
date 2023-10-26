#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:26:46 2023

@author: raquelgonzalezarmas
"""


"""
    ===> Script to find a satisfactory baseline simulation for LIAISE
"""

"""
Created on Wed Feb 15 13:17:29 2023

@author: raquelgonzalezarmas
"""

import sys
sys.path.append('/Users/raquelgonzalezarmas/OneDrive - Wageningen University & Research/CLASS_Raquel/modelpy/')
import model_V3 as CLASS
import model_cloud as CLASS_cloud
import numpy as np
sys.path.append('/Users/raquelgonzalezarmas/Library/CloudStorage/OneDrive-WageningenUniversity&Research/PhD_Wageningen/Code/Numerical-Methods')
import step_functions as step


""" 
Create empty model_input and set up case
"""



# La Cendrosa Local interactive

def create_input(run1input):
    # -> Switches (activation of certain parts of the model)
    run1input.sw_ml      = True          # mixed-layer model 
    run1input.sw_shearwe = False         # shear growth mixed-layer
    run1input.sw_fixft   = False         # Fix the free-troposphere 
    run1input.sw_wind    = False         # prognostic wind 
    run1input.sw_sl      = True          # surface layer 
    run1input.sw_rad     = True          # radiation 
    run1input.sw_ls      = True          # land surface 
    run1input.sw_cu      = False         # cumulus parameterization
    run1input.sw_gsdelay = False         # stomatal delay response switch
    
    # (1) Time variables
    run1input.dt         = 5.            # time step [s]
    run1input.runtime    = 16*3600.      # total run time [s]
    run1input.tstart     = 5.0           # time of the day [h UTC]
    
    # (2-3) Initialize mixed-layer characteristics and prognostic wind (swicth sw_ml and switch sw_wind)
    
    ## Input needed variables
    
    ## (2a) General boundary layer characteristics
    run1input.h          = 150.#150          # initial ABL height [m]
    run1input.Ps         = 101300.       # surface pressure [Pa]
    run1input.divU       = -0.           # horizontal large-scale divergence of wind [s-1]
    run1input.fc         = 1.e-4         # Coriolis parameter [m s-1]
    
    ## (2b) Temperature/Heat
    run1input.theta      = 293#292.5      # initial mixed-layer potential temperature [K]
    run1input.dtheta     = 1.5           # initial temperature jump at h [K]
    run1input.gammatheta = 0.012      # free atmosphere potential temperature lapse rate [K m-1]
    run1input.advtheta   = 0            # advection of heat [K s-1]
    run1input.beta       = 0.2          # entrainment ratio for virtual heat [-]
    run1input.wtheta     = 0.1       # surface kinematic heat flux [K m s-1]
    
    ## (2c) Specific humidity/Latent heat
    run1input.q          = 0.0095       # initial mixed-layer specific humidity [kg kg-1]
    run1input.dq         = -0.002       # initial specific humidity jump at h [kg kg-1] 
    run1input.gammaq     = -1e-5    # free atmosphere specific humidity lapse rate [kg kg-1 m-1] - good value
    run1input.advq       = 0.           # advection of moisture [kg kg-1 s-1]
    run1input.wq         =  0.000001    # surface kinematic moisture flux [kg kg-1 m s-1]
    
    ## (2d) CO2
    run1input.CO2        = 470.      # initial mixed-layer CO2 [ppm]
    run1input.dCO2       = -65.      # initial CO2 jump at h [ppm]
    run1input.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
    run1input.advCO2     = 0.        # advection of CO2 [ppm s-1]
    run1input.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]
    
    # (3) Initialize wind/momentum characteristics (sw_wind switch)
    
    run1input.u          = 2.        # initial mixed-layer u-wind speed [m s-1]
    run1input.du         = -2.2      # initial u-wind jump at h [m s-1]
    run1input.gammau     = 0.0       # free atmosphere u-wind speed lapse rate [s-1]
    run1input.advu       = 0         # advection of u-wind [m s-2]
        
    run1input.v          = 0.0       # initial mixed-layer u-wind speed [m s-1]
    run1input.dv         = 0.0       # initial u-wind jump at h [m s-1]
    run1input.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
    run1input.advv       = 0.        # advection of v-wind [m s-2]
    
    # (4) Initialize surface_layer characteristics (sw_sl switch)
    
    run1input.ustar      = 0.1       # surface friction velocity [m s-1]
    run1input.z0m        = 0.045     # roughness length for momentum [m]
    run1input.z0h        = 0.0045    # roughness length for scalars [m]
        
        
        
    # (5) Initialize radiation (switch sw_rad)
        
    run1input.lat        = 41.69336111  # latitude [deg]
    run1input.lon        = 0.928411111  # longitude [deg]
    run1input.doy        = 198.         # day of the year [-]
    run1input.cc         = 0.0          # cloud cover fraction [-]
    run1input.Q          = 0.           # net radiation [W m-2] 
    run1input.dFz        = 0.           # cloud top radiative divergence [W m-2] 
    
    
    # (6) Initialize land-surface (switch sw_ls)
    
    run1input.ls_type    = 'js'       # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
    run1input.c3c4       = 'c3'       # Plant type ('c3' or 'c4')
    
    run1input.wg         = 0.21       # volumetric water content top soil layer [m3 m-3]  
    run1input.w2         = 0.3        # volumetric water content deeper soil layer [m3 m-3]
    run1input.Tsoil      = 293.       #289.5    #after spin up- soil temp
    run1input.T2         = 289.       #290.5  # temperature deeper soil layer [K]
    
    run1input.a          = 0.219      # Clapp and Hornberger retention curve parameter a
    run1input.b          = 5.39       # Clapp and Hornberger retention curve parameter b
    run1input.p          = 4.         # Clapp and Hornberger retention curve parameter c
    run1input.CGsat      = 3.56e-6    # saturated soil conductivity for heat (K m-2 J-1)
    
    run1input.wsat       = 0.472      # saturated volumetric water content ECMWF config [-]
    run1input.wfc        = 0.323      # volumetric water content field capacity [-]
    run1input.wwilt      = 0.171      # volumetric water content wilting point [-]
    
    run1input.C1sat      = 0.132      # coefficent force term moisture 
    run1input.C2ref      = 1.8        # coefficent restore term moisture 
    
    run1input.c_beta     = 0.        # Curvature plant water-stress factor (0..1) [-]
    
    run1input.LAI        = 1.33       # leaf area index [-]
    run1input.gD         = 0.0        # correction factor transpiration for VPD [-]
    run1input.rsmin      = 110.       # minimum resistance transpiration [s m-1]
    run1input.rssoilmin  = 500.       # minimun resistance soil evaporation [s m-1]
    run1input.alpha      = 0.20       # surface albedo [-]
    run1input.rssoil     = 1.e6
    
    run1input.Ts         = 293#298.      # initial surface temperature [K] - not so senstive b/c calcs.
    
    run1input.cveg       = 1.         # vegetation fraction [-]
    run1input.Wmax       = 1#0.0002    # thickness of water layer on wet vegetation [m]
    run1input.Wl         = 1e-10#0.00014  # equivalent water layer depth for wet vegetation [m]
    run1input.cliq       = 0.        # wet fraction [-]
    
    run1input.Lambda     = 50#30.#20       # thermal diffusivity skin layer [-]
    
    # (7) Initialize cumulus parameterization
        
    run1input.sw_cu      = False     # Cumulus parameterization switch
    run1input.dz_h       = 150.      # Transition layer thickness [m]
    
    return run1input

def jarvis_control():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    theta_adv_cte = 5/(6*3600)     # K / s
    theta_adv = step.Flan(time, theta_adv_cte, 0.5, 11,18)
    q_adv_cte = -0.0003 * 1/(15*3600)                           # kg water/kg air /s
    #q_adv = q_adv_cte*np.ones(len(time))
    q_adv = step.Flan(time, q_adv_cte, 0.5, 11,18)
    # advection of wind     
    time_u = np.arange(5,22)
    time_u_new = (time_u-run1input.tstart) * 3600
    u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,theta_adv),
                            'advq':    (time_new-time_shift, q_adv),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1





def control():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_model_1_2 = (T_adv_model_1 + T_adv_model_2 )/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,22, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    #u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1

def control_ent_DA():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    run1input.dq = -0.004
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_model_1_2 = (T_adv_model_1 + T_adv_model_2 )/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,21, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    # jump of specific humidity
    time_dq = np.arange(5,21,(21-5)*3600/5)
    #dq = -0.002*np.ones(len(time_dq)) #- 0.005 * step.Flan(time_dq, 1, 10, 11,13)
    #time_dq_new = (time_dq-run1input.tstart) * 3600
    
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }#'dq': (time_dq_new, dq)
                            #}
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1

def control_advection_warm_air():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_model_1_2 = (T_adv_model_1 + T_adv_model_2 )/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,22, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    #u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1

def control_adv_CA():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_cte_3 = -3
    T_adv_model_3 = step.Flan_skewed(x = time, amplitude =T_adv_cte_3, k1 = 6, k2 = 1, a= 12, b= 14)
    T_adv_model_1_2_3 = (T_adv_model_1 + T_adv_model_2 + T_adv_model_3)/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,22, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    #u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2_3),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1

def control_cloud():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_model_1_2 = (T_adv_model_1 + T_adv_model_2 )/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,22, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    #u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS_cloud.model(run1input)
    r1.run()
    return r1

def control_cloud2():
    """
    Init and run the model
    """
    run1input = CLASS.model_input()
    run1input = create_input(run1input)
    # Modify LS scheme
    run1input.LAI = 1.
    run1input.ls_type = 'ags'
    # fitted parameters
    run1input.Ammax298[0] = 3.024
    run1input.gm298[0] = 10.2
    #run1input.gm298[0] = 8
    run1input.alpha0[0] = 0.0265
    run1input.T2gm[0] = 306
    # Add cloud cover
    run1input.cc = 0
    # Add advection
    time = np.arange(5,21,0.1)                                   # Hour in UTC
    time_new = (time-run1input.tstart) * 3600
    time_shift = 1/4*3600
    ## Input the boundary layer characteristics with time evolution of regional case
    ### Constants
    ### Model 2
    #### Heat
    T_adv_cte_1 = 0.8
    T_adv_model_1 = step.Flan(time, T_adv_cte_1, 1, 5,6.5)
    T_adv_cte_2 = 1
    T_adv_model_2 = step.Flan(time, T_adv_cte_2, 0.7, 10,16.5)
    T_adv_model_1_2 = (T_adv_model_1 + T_adv_model_2 )/3600      # K/s
    #### Moisture
    q_adv_cte_1 = -0.15
    q_adv_model_1 = step.Flan_skewed(x = time, amplitude = q_adv_cte_1, k1 = 3.9, k2 = 0.8, a = 5.2, b = 6)
    q_adv_cte_2 = -0.0003 * 1/(15)*1000   
    q_adv_model_2 = step.Flan(time, q_adv_cte_2, 0.5, 11,18)
    q_adv_model_1_2 = (q_adv_model_1 + q_adv_model_2 )/1000/3600 # kg water / kg air /s
    # advection of wind     
    time_u = np.arange(5,22, 0.05)
    p = np.array([ 6.60186316e-04, -3.60338270e-02,  7.35099740e-01, -6.54173806e+00, 2.45287069e+01])
    u = p[0]*time_u**4 + p[1]*time_u**3 + p[2]*time_u**2 + p[3]*time_u + p[4]
    time_u_new = (time_u-run1input.tstart) * 3600
    #u = np.array([5,6,5, 4, 3, 3., 3.4, 3.8,4.,4,4.5,4.3,4.,4,5,7,7])
    
    run1input.timedep =     {\
                            'advtheta':(time_new-time_shift,T_adv_model_1_2),
                            'advq':    (time_new-time_shift, q_adv_model_1_2),
                            'u':    (time_u_new, u),
                            }
    #run model: 
    r1 = CLASS.model(run1input)
    r1.run()
    return r1


