# 
# CLASS
# Copyright (c) 2010-2015 Meteorology and Air Quality section, Wageningen University and Research centre
# Copyright (c) 2011-2015 Jordi Vila-Guerau de Arellano
# Copyright (c) 2011-2015 Chiel van Heerwaarden
# Copyright (c) 2011-2015 Bart van Stratum
# Copyright (c) 2011-2015 Kees van den Dries
# 
# This file is part of CLASS
# 
# CLASS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# CLASS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CLASS.  If not, see <http://www.gnu.org/licenses/>.
#
# modified by Raquel:
# (1) Time-dependent variables can be input to the model
# (2) Correction of an equation of the A-gs model
# (3) Calculation of time spent in executing each package
# (4) A-gs parameters can be input in the modelled
# (5) Saving additional output variables: INDICATE THEM

import copy as cp
import numpy as np
import time
import sys
sys.path.append('/Users/raquelgonzalezarmas/Desktop/PhD_Wageningen/Code/Numerical-Methods/')
import Integration as integration
import step_functions as step
import xarray as xr

SWdown = xr.open_dataarray('/Users/raquelgonzalezarmas/Library/CloudStorage/'+\
                         'OneDrive-WageningenUniversity&Research/PhD_Wagen'+\
                             'ingen/1er-paper/observations/cloudcase/'+\
                                 'SWdown_cloud.nc')
#import ribtol

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

def qsat(T,p):
    return 0.622 * esat(T) / p

class model:
    def __init__(self, model_input):
        # initialize the different components of the model
        self.input = cp.deepcopy(model_input)
  
    def run(self):
        # initialize model variables
        self.init()
  
        # time integrate model 
        for self.t in range(self.tsteps):
            # time integrate components
            #print(self.t)
            self.timestep()
  
        # delete unnecessary variables from memory
        self.exitmodel()
    
    def init(self):
        # assign variables from input data
        # initialize constants
        self.Lv         = 2.5e6                 # heat of vaporization [J kg-1]
        self.cp         = 1005.                 # specific heat of dry air [J kg-1 K-1]
        self.rho        = 1.2                   # density of air [kg m-3]
        self.k          = 0.4                   # Von Karman constant [-]
        self.g          = 9.81                  # gravity acceleration [m s-2]
        self.Rd         = 287.                  # gas constant for dry air [J kg-1 K-1]
        self.Rv         = 461.5                 # gas constant for moist air [J kg-1 K-1]
        self.bolz       = 5.67e-8               # Bolzman constant [-]
        self.rhow       = 1000.                 # density of water [kg m-3]
        self.S0         = 1368.                 # solar constant [W m-2]

        # A-Gs constants and settings
        # Plant type:
        self.CO2comp298 =  self.input.CO2comp298   # CO2 compensation concentration [mg m-3]
        self.Q10CO2     =  self.input.Q10CO2       # function parameter to calculate CO2 compensation concentration [-]
        self.gm298      =  self.input.gm298        # mesophyill conductance at 298 K [mm s-1]
        self.Ammax298   =  self.input.Ammax298     # CO2 maximal primary productivity [mg m-2 s-1]
        self.Q10gm      =  self.input.Q10gm        # function parameter to calculate mesophyll conductance [-]
        self.T1gm       =  self.input.T1gm         # reference temperature to calculate mesophyll conductance gm [K]
        self.T2gm       =  self.input.T2gm         # reference temperature to calculate mesophyll conductance gm [K]
        self.Q10Am      =  self.input.Q10Am        # function parameter to calculate maximal primary profuctivity Ammax
        self.T1Am       =  self.input.T1Am         # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.T2Am       =  self.input.T2Am         # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.f0         =  self.input.f0           # maximum value Cfrac [-]
        self.ad         =  self.input.ad           # regression coefficient to calculate Cfrac [kPa-1]
        self.alpha0     =  self.input.alpha0       # initial low light conditions [mg J-1]
        self.Kx         =  self.input.Kx           # extinction coefficient PAR [-]
        self.gmin       =  self.input.gmin         # cuticular (minimum) conductance [m s-1]
        
        # Farquhar-vonCaemmerer-Berry and Medlyn parameters
        self.Vcmax298   = self.input.Vcmax298      # maximum velocities of the carboxylase at 298 K (mol CO2/m2/s)
        self.Jmax298    = self.input.Jmax298       # maximum electron transport rate at 298 K (mol electrons /m2/s)
        self.Kc298      = self.input.Kc298         # Michaelis-Menten constant for CO2 at 298 K (mol CO2/ mol air)
        self.Ko298      = self.input.Ko298         # Michaelis-Menten constant for O2 at 298 K (mol O2/mol air)
        self.f_         = self.input.f_            # Fraction of light lost as absorption by other than the chloroplast lamellae (-)
        self.CO2comp298_= self.input.CO2comp298_   # Compensation point at 298 K
        self.Oi         = self.input.Oi            # Leaf interior oxygen concentration (mol O2/mol air)
        self.g0_        = self.input.g0_           # minimum stomtatal conductance (mol CO2/m2/s)
        self.g1_        = self.input.g1_           # parameter to calculate stomata conductance (Pa-1/2)
        self.Kx_        = self.input.Kx_           # extinction coefficient PAR [-]
        
        self.mco2       =  44.;                 # molecular weight CO2 [g mol -1]
        self.mair       =  28.9;                # molecular weight air [g mol -1]
        self.nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide 

        self.Cw         =  0.0016;              # constant water stress correction (eq. 13 Jacobs et al. 2007) [-]
        self.wmax       =  0.55;                # upper reference value soil water [-]
        self.wmin       =  0.005;               # lower reference value soil water [-]
        self.R10        =  0.1#0.15#0.23;                # respiration at 10 C [mg CO2 m-2 s-1]
        self.E0         =  53.3e3;              # activation energy [53.3 kJ kmol-1]

        # Read switches
        self.sw_ml      = self.input.sw_ml      # mixed-layer model switch
        self.sw_shearwe = self.input.sw_shearwe # shear growth ABL switch
        self.sw_fixft   = self.input.sw_fixft   # Fix the free-troposphere switch
        self.sw_wind    = self.input.sw_wind    # prognostic wind switch
        self.sw_sl      = self.input.sw_sl      # surface layer switch
        self.sw_rad     = self.input.sw_rad     # radiation switch
        self.sw_ls      = self.input.sw_ls      # land surface switch
        self.ls_type    = self.input.ls_type    # land surface paramaterization ('js' or 'ags' or 'FvCB_M')
        self.sw_gsdelay = self.input.sw_gsdelay # stomatal conductance delay switch(True or False)
        self.sw_cu      = self.input.sw_cu      # cumulus parameterization switch
  
        # initialize mixed-layer
        self.h          = self.input.h          # initial ABL height [m]
        self.Ps         = self.input.Ps         # surface pressure [Pa]
        self.divU       = self.input.divU       # horizontal large-scale divergence of wind [s-1]
        self.ws         = None                  # large-scale vertical velocity [m s-1]
        self.wf         = None                  # mixed-layer growth due to radiative divergence [m s-1]
        self.fc         = self.input.fc         # coriolis parameter [s-1]
        self.we         = -1.                   # entrainment velocity [m s-1]
       
         # Temperature 
        self.theta      = self.input.theta      # initial mixed-layer potential temperature [K]
        self.dtheta     = self.input.dtheta     # initial temperature jump at h [K]
        self.gammatheta = self.input.gammatheta # free atmosphere potential temperature lapse rate [K m-1]
        self.advtheta   = self.input.advtheta   # advection of heat [K s-1]
        self.beta       = self.input.beta       # entrainment ratio for virtual heat [-]
        self.wtheta     = self.input.wtheta     # surface kinematic heat flux [K m s-1]
        self.wthetae    = None                  # entrainment kinematic heat flux [K m s-1]
 
        self.wstar      = 0.                    # convective velocity scale [m s-1]
 
        # 2m diagnostic variables 
        self.T2m        = None                  # 2m temperature [K]
        self.q2m        = None                  # 2m specific humidity [kg kg-1]
        self.e2m        = None                  # 2m vapor pressure [Pa]
        self.esat2m     = None                  # 2m saturated vapor pressure [Pa]
        self.u2m        = None                  # 2m u-wind [m s-1]
        self.v2m        = None                  # 2m v-wind [m s-1]
 
        # 0.105 m diagnostic variables
        self.T0_105m    = None                  # 0.105m temperature [K]
        self.q0_105m    = None                  # 0.105m specific humidity [kg kg-1]
        self.e0_105m    = None                  # 0.105m vapor pressure [Pa]
        self.esat0_105m = None                  # 0.105m saturated vapor pressure [Pa]
        self.u0_105m    = None                  # 0.105m u-wind [m s-1]
        self.v0_105m    = None                  # 0.105m v-wind [m s-1]
        
        
        # Surface variables 
        self.thetasurf  = self.input.theta      # surface potential temperature [K]
        self.thetavsurf = None                  # surface virtual potential temperature [K]
        self.qsurf      = None                  # surface specific humidity [g kg-1]

        # Mixed-layer top variables
        self.P_h        = None                  # Mixed-layer top pressure [pa]
        self.T_h        = None                  # Mixed-layer top absolute temperature [K]
        self.q2_h       = None                  # Mixed-layer top specific humidity variance [kg2 kg-2]
        self.CO22_h     = None                  # Mixed-layer top CO2 variance [ppm2]
        self.RH_h       = None                  # Mixed-layer top relavtive humidity [-]
        self.dz_h       = None                  # Transition layer thickness [m]
        self.lcl        = None                  # Lifting condensation level [m]

        # Virtual temperatures and fluxes
        self.thetav     = None                  # initial mixed-layer potential temperature [K]
        self.dthetav    = None                  # initial virtual temperature jump at h [K]
        self.wthetav    = None                  # surface kinematic virtual heat flux [K m s-1]
        self.wthetave   = None                  # entrainment kinematic virtual heat flux [K m s-1]
       
        # Moisture 
        self.q          = self.input.q          # initial mixed-layer specific humidity [kg kg-1]
        self.dq         = self.input.dq         # initial specific humidity jump at h [kg kg-1]
        self.gammaq     = self.input.gammaq     # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.advq       = self.input.advq       # advection of moisture [kg kg-1 s-1]
        self.wq         = self.input.wq         # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wqe        = None                  # entrainment moisture flux [kg kg-1 m s-1]
        self.wqM        = None                  # moisture cumulus mass flux [kg kg-1 m s-1]
  
        self.qsat       = None                  # mixed-layer saturated specific humidity [kg kg-1]
        self.esat       = None                  # mixed-layer saturated vapor pressure [Pa]
        self.e          = None                  # mixed-layer vapor pressure [Pa]
        self.qsatsurf   = None                  # surface saturated specific humidity [g kg-1]
        self.dqsatdT    = None                  # slope saturated specific humidity curve [g kg-1 K-1]
      
        # CO2
        fac = self.mair / (self.rho*self.mco2)  # Conversion factor mgC m-2 s-1 to ppm m s-1
        self.CO2        = self.input.CO2        # initial mixed-layer CO2 [ppm]
        self.dCO2       = self.input.dCO2       # initial CO2 jump at h [ppm]
        self.gammaCO2   = self.input.gammaCO2   # free atmosphere CO2 lapse rate [ppm m-1]
        self.advCO2     = self.input.advCO2     # advection of CO2 [ppm s-1]
        self.wCO2       = self.input.wCO2 * fac # surface kinematic CO2 flux [ppm m s-1]
        self.wCO2A      = 0                     # surface assimulation CO2 flux [ppm m s-1]
        self.wCO2R      = 0                     # surface respiration CO2 flux [ppm m s-1]
        self.wCO2e      = 0                     # entrainment CO2 flux [ppm m s-1]
        self.wCO2M      = 0                     # CO2 mass flux [ppm m s-1]
       
        # Wind 
        self.u          = self.input.u          # initial mixed-layer u-wind speed [m s-1]
        self.du         = self.input.du         # initial u-wind jump at h [m s-1]
        self.gammau     = self.input.gammau     # free atmosphere u-wind speed lapse rate [s-1]
        self.advu       = self.input.advu       # advection of u-wind [m s-2]
        
        self.v          = self.input.v          # initial mixed-layer u-wind speed [m s-1]
        self.dv         = self.input.dv         # initial u-wind jump at h [m s-1]
        self.gammav     = self.input.gammav     # free atmosphere v-wind speed lapse rate [s-1]
        self.advv       = self.input.advv       # advection of v-wind [m s-2]
        
        # Tendencies 
        self.htend      = None                  # tendency of CBL [m s-1]
        self.thetatend  = None                  # tendency of mixed-layer potential temperature [K s-1]
        self.dthetatend = None                  # tendency of potential temperature jump at h [K s-1]
        self.qtend      = None                  # tendency of mixed-layer specific humidity [kg kg-1 s-1]
        self.dqtend     = None                  # tendency of specific humidity jump at h [kg kg-1 s-1]
        self.CO2tend    = None                  # tendency of CO2 humidity [ppm]
        self.dCO2tend   = None                  # tendency of CO2 jump at h [ppm s-1]
        self.utend      = None                  # tendency of u-wind [m s-1 s-1]
        self.dutend     = None                  # tendency of u-wind jump at h [m s-1 s-1]
        self.vtend      = None                  # tendency of v-wind [m s-1 s-1]
        self.dvtend     = None                  # tendency of v-wind jump at h [m s-1 s-1]
        self.dztend     = None                  # tendency of transition layer thickness [m s-1]
  
        # initialize surface layer
        self.ustar      = self.input.ustar      # surface friction velocity [m s-1]
        self.uw         = None                  # surface momentum flux in u-direction [m2 s-2]
        self.vw         = None                  # surface momentum flux in v-direction [m2 s-2]
        self.z0m        = self.input.z0m        # roughness length for momentum [m]
        self.z0h        = self.input.z0h        # roughness length for scalars [m]
        self.Cm         = 1e12                  # drag coefficient for momentum [-]
        self.Cs         = 1e12                  # drag coefficient for scalars [-]
        self.L          = None                  # Obukhov length [m]
        self.Rib        = None                  # bulk Richardson number [-]
        self.ra         = None                  # aerodynamic resistance [s m-1]
  
        # initialize radiation
        self.lat        = self.input.lat        # latitude [deg]
        self.lon        = self.input.lon        # longitude [deg]
        self.doy        = self.input.doy        # day of the year [-]
        self.tstart     = self.input.tstart     # time of the day [-]
        self.cc         = self.input.cc         # cloud cover fraction [-]
        self.Swin       = None                  # incoming short wave radiation [W m-2]
        self.Swout      = None                  # outgoing short wave radiation [W m-2]
        self.Lwin       = None                  # incoming long wave radiation [W m-2]
        self.Lwout      = None                  # outgoing long wave radiation [W m-2]
        self.Q          = self.input.Q          # net radiation [W m-2]
        self.dFz        = self.input.dFz        # cloud top radiative divergence [W m-2] 
  
        # initialize land surface
        self.wg         = self.input.wg         # volumetric water content top soil layer [m3 m-3]
        self.w2         = self.input.w2         # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil      = self.input.Tsoil      # temperature top soil layer [K]
        self.T2         = self.input.T2         # temperature deeper soil layer [K]
                           
        self.a          = self.input.a          # Clapp and Hornberger retention curve parameter a [-]
        self.b          = self.input.b          # Clapp and Hornberger retention curve parameter b [-]
        self.p          = self.input.p          # Clapp and Hornberger retention curve parameter p [-]
        self.CGsat      = self.input.CGsat      # saturated soil conductivity for heat
                           
        self.wsat       = self.input.wsat       # saturated volumetric water content ECMWF config [-]
        self.wfc        = self.input.wfc        # volumetric water content field capacity [-]
        self.wwilt      = self.input.wwilt      # volumetric water content wilting point [-]
                           
        self.C1sat      = self.input.C1sat      
        self.C2ref      = self.input.C2ref      

        self.c_beta     = self.input.c_beta     # Curvature plant water-stress factor (0..1) [-]
        
        self.LAI        = self.input.LAI        # leaf area index [-]
        self.gD         = self.input.gD         # correction factor transpiration for VPD [-]
        self.rsmin      = self.input.rsmin      # minimum resistance transpiration [s m-1]
        self.rssoilmin  = self.input.rssoilmin  # minimum resistance soil evaporation [s m-1]
        self.alpha      = self.input.alpha      # surface albedo [-]
  
        self.rs         = 1.e6                  # resistance transpiration [s m-1]
        self.rssoil     = 1.e6                  # resistance soil [s m-1]
                           
        self.Ts         = self.input.Ts         # surface temperature [K]
                           
        self.cveg       = self.input.cveg       # vegetation fraction [-]
        self.Wmax       = self.input.Wmax       # thickness of water layer on wet vegetation [m]
        self.Wl         = self.input.Wl         # equivalent water layer depth for wet vegetation [m]
        self.cliq       = None                  # wet fraction [-]
                          
        self.Lambda     = self.input.Lambda     # thermal diffusivity skin layer [-]
  
        self.Tsoiltend  = None                  # soil temperature tendency [K s-1]
        self.wgtend     = None                  # soil moisture tendency [m3 m-3 s-1]
        self.Wltend     = None                  # equivalent liquid water tendency [m s-1]
  
        self.H          = None                  # sensible heat flux [W m-2]
        self.LE         = None                  # evapotranspiration [W m-2]
        self.LEliq      = None                  # open water evaporation [W m-2]
        self.LEveg      = None                  # transpiration [W m-2]
        self.LEsoil     = None                  # soil evaporation [W m-2]
        self.LEpot      = None                  # potential evaporation [W m-2]
        self.LEref      = None                  # reference evaporation using rs = rsmin / LAI [W m-2]
        self.G          = None                  # ground heat flux [W m-2]
        
        
        # Raquel's additions 
        self.ETcommonpart = None                # common part of soil, vegetation and liquid ET [W/m/s]
        self.R            = None                # Total resistance of an ecosystem [s m-1]
        self.LE_radiation = None                # LE part related to available energy [W/m2]
        self.LE_VPD       = None                # LE part related to the VPD [W/m2]
        self.LE_ideal     = None                # LE part taking into account qsatsurf (no EBC explicitly assumption) [W/m2]
        self.gveg_eff     = None                # Effective vegetation conductance [m/s]
        self.gsoil_eff    = None                # Effective soil conductance [m/s]
        self.gliq_eff     = None                # Effective liquid conductance [m/s]
        self.D0           = None                # Vapor pressure deficit when stomata close [Pa]
        self.Ds           = None                # Leaf to air vapor pressure deficit [kPa]
        self.PAR          = None                # Photosynthetic active radiation (W/m2)
        # ags
        self.co2abs       = None                # atmospheric CO2 concentration (mg CO2/m3)
        self.ci           = None                # CO2 concentration in the interior of the plant (mg CO2/m3)
        self.Ammax        = None                # CO2 maximal primary production (mg/m2leaf/s)
        self.Am           = None                # CO2 primary productivity (mg/m2leaf/s)
        self.Rdark        = None                # Dark respiration (mg/m2leaf/s)
        self.Ag           = None                # CO2 gross primary productivity at leaf level (mg/m2leaf/s)
        self.Agstar       = None                # Soil water unstressed CO2 gross primary productivity at leaf level (mg/m2leaf/s)
        self.Ag1          = None                # CO2 gross primary productivity at canopy (mg/m2/s)
        self.An           = None                # net flux of CO2 into the plant (mg/m2/s) 
        self.PCbeta       = None                # Plant water stress function to calculate factor (intermediate step)
        self.fstr         = None                # Plant water stress factor (0..1) [-]
        self.glw          = None                # Stomatal conductance for water at leaf level [m/s]
        self.glc          = None                # Stomatal conductance for carbon dioxide at leaf level [m/s]
        self.glc2_3       = None                # Stomatal conductance for carbon dioxide at leaf level [m/s] for leaf at 2/3 of canopy height
        self.glc1_3       = None                # Stomatal conductance for carbon dioxide at leaf level [m/s] for leaf at 1/3 of canopy height
        self.glc0         = None                # Stomatal conductance for carbon dioxide at leaf level [m/s] for leaf at 0 canopy height (leaf in the ground)
        self.LEleaf       = None                # Latent heat at leaf level [UNITS!!]
        self.taug         = 15*60               # Time constant that characterizes stomatal delay response (s)
        ## to disentangle diurnal variability of glc
        self.a1           = None                # Intermediate parameter to calculate stomata conductance (-)
        self.alphac       = None                # Light use efficiency (mg J-1)
        self.CO2comp      = None                # CO2 compensation point (mg/m3)
        self.Dstar        = None                # Intermediate variable to calculate stomata conductance (Pa)
        self.fmin         = None                # Minimum value of Cfrac (-)
        self.fmin0        = None                # Intermediate conductance used to derive fmin (m/s)
        self.gm           = None                # Mesophyll conductance (m/s)
        self.cfrac        = None                # Fraction of the concentrations (ci-gamma)/(co2abs-gamma) (-)
        self.Aln          = None                # Net flow of CO2 into the plant per leaf area (mg CO2/m^2leaf/s)
        ### Q: I think that according to theory Ag1 should be equal to An but I will check the difference. In the code they seem to be different
        
        
        
        self.Resp         = None                # CO2 soil surface flux (mg/m2/s)
        # End of Raquel's additions
        
        # initialize A-Gs surface scheme
        self.c3c4       = self.input.c3c4       # plant type ('c3' or 'c4')
 
        # initialize cumulus parameterization
        self.sw_cu      = self.input.sw_cu
        self.dz_h       = self.input.dz_h
        self.ac         = 0.                    # Cloud core fraction [-]
        self.M          = 0.                    # Cloud core mass flux [m s-1] 
        self.wqM        = 0.                    # Cloud core moisture flux [kg kg-1 m s-1] 

        # Time or height dependent properties (currently both time and height dependency is not supported)
        self.timedep    = self.input.timedep    # Time dependent variables
        self.heightdep  = self.input.heightdep  # Height dependent variables 
  
        # initialize time variables
        self.tsteps = int(np.floor(self.input.runtime / self.input.dt))
        self.dt     = self.input.dt
        self.t      = 0
        
        # Some sanity checks for valid input
        if (self.c_beta is None): 
            self.c_beta = 0                     # Zero curvature; linear response
        assert(self.c_beta >= 0 or self.c_beta <= 1)
        
        # initialize output
        self.out = model_output(self.tsteps)
        
        # Update height or time dependent variables
        self.update_time_height_dependence()
        
        self.statistics()
  
        # calculate initial diagnostic variables
        if(self.sw_rad):
            self.run_radiation()
 
        if(self.sw_sl):
            for i in range(10): 
                self.run_surface_layer()
  
        if(self.sw_ls):
            self.run_land_surface()

        if(self.sw_cu):
            self.run_mixed_layer()
            self.run_cumulus()
        
        if(self.sw_ml):
            self.run_mixed_layer()
        
    def timestep(self):
        
        # Update height or time dependent variables
        self.update_time_height_dependence()
        
        self.statistics()

        # run radiation model
        if(self.sw_rad):
            self.run_radiation()
            
        # run surface layer model
        if(self.sw_sl):
            self.run_surface_layer()
        
        # run land surface model
        if(self.sw_ls):
            self.run_land_surface()
 
        # run cumulus parameterization
        if(self.sw_cu):
            self.run_cumulus()
   
        # run mixed-layer model
        if(self.sw_ml):
            self.run_mixed_layer()
 
        # store output before time integration
        self.store()
  
        # time integrate land surface model
        if(self.sw_ls):
            self.integrate_land_surface()
  
        # time integrate mixed-layer model
        if(self.sw_ml):
            self.integrate_mixed_layer()
  
    def statistics(self):
        # Calculate virtual temperatures 
        self.thetav   = self.theta  + 0.61 * self.theta * self.q
        self.wthetav  = self.wtheta + 0.61 * self.theta * self.wq
        self.dthetav  = (self.theta + self.dtheta) * (1. + 0.61 * (self.q + self.dq)) - self.theta * (1. + 0.61 * self.q)

        # Mixed-layer top properties
        self.P_h    = self.Ps - self.rho * self.g * self.h
        self.T_h    = self.theta - self.g/self.cp * self.h

        #self.P_h    = self.Ps / np.exp((self.g * self.h)/(self.Rd * self.theta))
        #self.T_h    = self.theta / (self.Ps / self.P_h)**(self.Rd/self.cp)

        self.RH_h   = self.q / qsat(self.T_h, self.P_h)

        # Find lifting condensation level iteratively
        if(self.t == 0):
            self.lcl = self.h
            RHlcl = 0.5
        else:
            RHlcl = 0.9998 

        itmax = 30
        it = 0
        while(((RHlcl <= 0.9999) or (RHlcl >= 1.0001)) and it<itmax):
            self.lcl    += (1.-RHlcl)*1000.
            p_lcl        = self.Ps - self.rho * self.g * self.lcl
            T_lcl        = self.theta - self.g/self.cp * self.lcl
            RHlcl        = self.q / qsat(T_lcl, p_lcl)
            it          += 1

        if(it == itmax):
            print("LCL calculation not converged!!")
            print("RHlcl = %f, zlcl=%f"%(RHlcl, self.lcl))

    def update_time_height_dependence(self):
        # Update time varying variables
        if self.timedep is not None:
            time = self.t*self.dt
            for name, var_tupple in self.timedep.items():
                # check if `name` is a class variable which we can update:
                if hasattr(self, name):
                    # Extract arrays with times and values from input, and interpolate:
                    times   = var_tupple[0]
                    values  = var_tupple[1]
                    new_val = np.interp(time, times, values)

                    # Update class variable
                    setattr(self, name, new_val)
                else:
                    print('Can not update time dependent variable {}!'.format(name))

        # Update height varying variables
        if self.heightdep is not None:
            for name, var_tupple in self.heightdep.items():
                # check if `name` is a class variable which we can update:
                if hasattr(self, name):
                    # Extract arrays with heights and values from input, and interpolate (binned):
                    heights = var_tupple[0]
                    values  = var_tupple[1]
                    index   = np.abs(heights - self.h).argmin()
                    if heights[index] > self.h:
                        index -= 1
                    new_val = values[index]

                    # Update class variable
                    setattr(self, name, new_val)
                else:
                    print('Can not update height dependent variable {}!'.format(name))
                    

    def run_cumulus(self):
        # Calculate mixed-layer top relative humidity variance (Neggers et. al 2006/7)
        if(self.wthetav > 0):
            self.q2_h   = -(self.wqe  + self.wqM  ) * self.dq   * self.h / (self.dz_h * self.wstar)
            self.CO22_h = -(self.wCO2e+ self.wCO2M) * self.dCO2 * self.h / (self.dz_h * self.wstar)
        else:
            self.q2_h   = 0.
            self.CO22_h = 0.

        # calculate cloud core fraction (ac), mass flux (M) and moisture flux (wqM)
        self.ac     = max(0., 0.5 + (0.36 * np.arctan(1.55 * ((self.q - qsat(self.T_h, self.P_h)) / self.q2_h**0.5))))
        self.M      = self.ac * self.wstar
        self.wqM    = self.M * self.q2_h**0.5

        # Only calculate CO2 mass-flux if mixed-layer top jump is negative
        if(self.dCO2 < 0):
            self.wCO2M  = self.M * self.CO22_h**0.5
        else:
            self.wCO2M  = 0.

    def run_mixed_layer(self):
        if(not self.sw_sl):
            # decompose ustar along the wind components
            self.uw = - np.sign(self.u) * (self.ustar ** 4. / (self.v ** 2. / self.u ** 2. + 1.)) ** (0.5)
            self.vw = - np.sign(self.v) * (self.ustar ** 4. / (self.u ** 2. / self.v ** 2. + 1.)) ** (0.5)
      
        # calculate large-scale vertical velocity (subsidence)
        self.ws = -self.divU * self.h
      
        # calculate compensation to fix the free troposphere in case of subsidence 
        if(self.sw_fixft):
            w_th_ft  = self.gammatheta * self.ws
            w_q_ft   = self.gammaq     * self.ws
            w_CO2_ft = self.gammaCO2   * self.ws 
        else:
            w_th_ft  = 0.
            w_q_ft   = 0.
            w_CO2_ft = 0. 
      
        # calculate mixed-layer growth due to cloud top radiative divergence
        self.wf = self.dFz / (self.rho * self.cp * self.dtheta)
       
        # calculate convective velocity scale w* 
        if(self.wthetav > 0.):
            self.wstar = ((self.g * self.h * self.wthetav) / self.thetav)**(1./3.)
        else:
            self.wstar  = 1e-6;
      
        # Virtual heat entrainment flux 
        self.wthetave    = -self.beta * self.wthetav 
        
        # compute mixed-layer tendencies
        if(self.sw_shearwe):
            self.we    = (-self.wthetave + 5. * self.ustar ** 3. * self.thetav / (self.g * self.h)) / self.dthetav
        else:
            self.we    = -self.wthetave / self.dthetav

        # Don't allow boundary layer shrinking if wtheta < 0 
        if(self.we < 0):
            self.we = 0.

        # Calculate entrainment fluxes
        self.wthetae     = -self.we * self.dtheta
        self.wqe         = -self.we * self.dq
        self.wCO2e       = -self.we * self.dCO2
  
        self.htend       = self.we + self.ws + self.wf - self.M
       
        self.thetatend   = (self.wtheta - self.wthetae             ) / self.h + self.advtheta 
        self.qtend       = (self.wq     - self.wqe     - self.wqM  ) / self.h + self.advq
        self.CO2tend     = (self.wCO2   - self.wCO2e   - self.wCO2M) / self.h + self.advCO2
        
        self.dthetatend  = self.gammatheta * (self.we + self.wf - self.M) - self.thetatend + w_th_ft
        self.dqtend      = self.gammaq     * (self.we + self.wf - self.M) - self.qtend     + w_q_ft
        self.dCO2tend    = self.gammaCO2   * (self.we + self.wf - self.M) - self.CO2tend   + w_CO2_ft
     
        # assume u + du = ug, so ug - u = du
        if(self.sw_wind):
            self.utend       = -self.fc * self.dv + (self.uw + self.we * self.du)  / self.h + self.advu
            self.vtend       =  self.fc * self.du + (self.vw + self.we * self.dv)  / self.h + self.advv
  
            self.dutend      = self.gammau * (self.we + self.wf - self.M) - self.utend
            self.dvtend      = self.gammav * (self.we + self.wf - self.M) - self.vtend
        
        # tendency of the transition layer thickness
        if(self.ac > 0 or self.lcl - self.h < 300):
            self.dztend = ((self.lcl - self.h)-self.dz_h) / 7200.
        else:
            self.dztend = 0.
   
    def integrate_mixed_layer(self):
        # set values previous time step
        h0      = self.h
        
        theta0  = self.theta
        dtheta0 = self.dtheta
        q0      = self.q
        dq0     = self.dq
        CO20    = self.CO2
        dCO20   = self.dCO2
        
        u0      = self.u
        du0     = self.du
        v0      = self.v
        dv0     = self.dv

        dz0     = self.dz_h
  
        # integrate mixed-layer equations
        self.h        = h0      + self.dt * self.htend
        self.theta    = theta0  + self.dt * self.thetatend
        self.dtheta   = dtheta0 + self.dt * self.dthetatend
        self.q        = q0      + self.dt * self.qtend
        self.dq       = dq0     + self.dt * self.dqtend
        self.CO2      = CO20    + self.dt * self.CO2tend
        self.dCO2     = dCO20   + self.dt * self.dCO2tend
        self.dz_h     = dz0     + self.dt * self.dztend

        # Limit dz to minimal value
        dz0 = 50
        if(self.dz_h < dz0):
            self.dz_h = dz0 
  
        if(self.sw_wind):
            self.u        = u0      + self.dt * self.utend
            self.du       = du0     + self.dt * self.dutend
            self.v        = v0      + self.dt * self.vtend
            self.dv       = dv0     + self.dt * self.dvtend
 
    def run_radiation(self):
        sda    = 0.409 * np.cos(2. * np.pi * (self.doy - 173.) / 365.)
        sinlea = np.sin(2. * np.pi * self.lat / 360.) * np.sin(sda) - np.cos(2. * np.pi * self.lat / 360.) * np.cos(sda) * np.cos(2. * np.pi * (self.t * self.dt + self.tstart * 3600.) / 86400. + 2. * np.pi * self.lon / 360.)
        sinlea = max(sinlea, 0.0001)
        
        Ta  = self.theta * ((self.Ps - 0.1 * self.h * self.rho * self.g) / self.Ps ) ** (self.Rd / self.cp)
  
        Tr  = (0.6 + 0.2 * sinlea) * (1. - 0.4 * self.cc)
  
        self.Swin  = self.S0 * Tr * sinlea
        
        t_hours = self.t*self.input.dt/3600
        
        self.Swin = self.Swin - step.Flan_skewed(t_hours, 279.2, 1.5, 1.5, 11-5, 13-5)
        #self.Swin = self.Swin - step.Flan_skewed(t_hours, 279.2, 1.467, 0.723, 12.4-5, 15.604-5)
        #self.Swin = SWdown.isel(datetime = self.t).item()
        """
        if (self.t>=6*3600/self.input.dt)&(self.t<=8*3600/self.input.dt):
            self.Swin = 600
        """
        self.Swout = self.alpha * self.S0 * Tr * sinlea
        self.Lwin  = 0.8 * self.bolz * Ta ** 4.
        self.Lwout = self.bolz * self.Ts ** 4.
          
        self.Q     = self.Swin - self.Swout + self.Lwin - self.Lwout
  
    def run_surface_layer(self):
        ueff           = max(0.01, np.sqrt(self.u**2. + self.v**2. + self.wstar**2.))
        self.thetasurf = self.theta + self.wtheta / (self.Cs * ueff)
        qsatsurf       = qsat(self.thetasurf, self.Ps)
        cq             = (1. + self.Cs * ueff * self.rs) ** -1.
        self.qsurf     = (1. - cq) * self.q + cq * qsatsurf

        self.thetavsurf = self.thetasurf * (1. + 0.61 * self.qsurf)
  
        zsl       = 0.1 * self.h
        self.Rib  = self.g / self.thetav * zsl * (self.thetav - self.thetavsurf) / ueff**2.
        self.Rib  = min(self.Rib, 0.2)

        self.L     = self.ribtol(self.Rib, zsl, self.z0m, self.z0h)  # Slow python iteration
        #self.L    = ribtol.ribtol(self.Rib, zsl, self.z0m, self.z0h) # Fast C++ iteration
        self.Cm   = self.k**2. / (np.log(zsl / self.z0m) - self.psim(zsl / self.L) + self.psim(self.z0m / self.L)) ** 2.
        self.Cs   = self.k**2. / (np.log(zsl / self.z0m) - self.psim(zsl / self.L) + self.psim(self.z0m / self.L)) / (np.log(zsl / self.z0h) - self.psih(zsl / self.L) + self.psih(self.z0h / self.L))
  
        self.ustar = np.sqrt(self.Cm) * ueff
        self.uw    = - self.Cm * ueff * self.u
        self.vw    = - self.Cm * ueff * self.v
 
        # diagnostic meteorological variables
        #labels_InCanopy = ['0.105 m', '0.29 m', '0.61 m', '0.99 m', '1.49 m']
        z_m = 2    # Height in m of T2m, q2m, u2m, v2m, esat2m, e2m
        self.T2m    = self.thetasurf - self.wtheta / self.ustar / self.k * (np.log(z_m / self.z0h) - self.psih(z_m / self.L) + self.psih(self.z0h / self.L))
        self.q2m    = self.qsurf     - self.wq     / self.ustar / self.k * (np.log(z_m / self.z0h) - self.psih(z_m / self.L) + self.psih(self.z0h / self.L))
        self.u2m    =                - self.uw     / self.ustar / self.k * (np.log(z_m / self.z0m) - self.psim(z_m / self.L) + self.psim(self.z0m / self.L))
        self.v2m    =                - self.vw     / self.ustar / self.k * (np.log(z_m / self.z0m) - self.psim(z_m / self.L) + self.psim(self.z0m / self.L))
        self.esat2m = 0.611e3 * np.exp(17.2694 * (self.T2m - 273.16) / (self.T2m - 35.86))
        self.e2m    = self.q2m * self.Ps / 0.622
        
        z_2 = 0.29     # Height in m of T2m, q2m, u2m, v2m, esat2m, e2m
        self.T0_105m    = self.thetasurf - self.wtheta / self.ustar / self.k * (np.log(z_2 / self.z0h) - self.psih(z_2 / self.L) + self.psih(self.z0h / self.L))
        self.q0_105m    = self.qsurf     - self.wq     / self.ustar / self.k * (np.log(z_2 / self.z0h) - self.psih(z_2 / self.L) + self.psih(self.z0h / self.L))
        self.u0_105m    =                - self.uw     / self.ustar / self.k * (np.log(z_2 / self.z0m) - self.psim(z_2 / self.L) + self.psim(self.z0m / self.L))
        self.v0_105m    =                - self.vw     / self.ustar / self.k * (np.log(z_2 / self.z0m) - self.psim(z_2 / self.L) + self.psim(self.z0m / self.L))
        self.esat0_105m = 0.611e3 * np.exp(17.2694 * (self.T0_105m - 273.16) / (self.T0_105m - 35.86))
        self.e0_105m    = self.q0_105m * self.Ps / 0.622
     
    def ribtol(self, Rib, zsl, z0m, z0h): 
        if(Rib > 0.):
            L    = 1.
            L0   = 2.
        else:
            L  = -1.
            L0 = -2.
        while (abs(L - L0) > 0.001):
            L0      = L
            fx      = Rib - zsl / L * (np.log(zsl / z0h) - self.psih(zsl / L) + self.psih(z0h / L)) / (np.log(zsl / z0m) - self.psim(zsl / L) + self.psim(z0m / L))**2.
            Lstart  = L - 0.001*L
            Lend    = L + 0.001*L
            fxdif   = ( (- zsl / Lstart * (np.log(zsl / z0h) - self.psih(zsl / Lstart) + self.psih(z0h / Lstart)) / \
                                          (np.log(zsl / z0m) - self.psim(zsl / Lstart) + self.psim(z0m / Lstart))**2.) \
                      - (-zsl /  Lend   * (np.log(zsl / z0h) - self.psih(zsl / Lend  ) + self.psih(z0h / Lend  )) / \
                                          (np.log(zsl / z0m) - self.psim(zsl / Lend  ) + self.psim(z0m / Lend  ))**2.) ) / (Lstart - Lend)
            L       = L - fx / fxdif
            if(abs(L) > 1e15):
                break
        return L
      
    def psim(self, zeta):
        if(zeta <= 0):
            x     = (1. - 16. * zeta)**(0.25)
            psim  = 3.14159265 / 2. - 2. * np.arctan(x) + np.log((1. + x)**2. * (1. + x**2.) / 8.)
            #x     = (1. + 3.6 * abs(zeta) ** (2./3.)) ** (-0.5)
            #psim = 3. * np.log( (1. + 1. / x) / 2.)
        else:
            psim  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - zeta - (10./3.) / 0.35
        return psim
      
    def psih(self, zeta):
        if(zeta <= 0):
            x     = (1. - 16. * zeta)**(0.25)
            psih  = 2. * np.log( (1. + x*x) / 2.)
            #x     = (1. + 7.9 * abs(zeta) ** (2./3.)) ** (-0.5)
            #psih  = 3. * np.log( (1. + 1. / x) / 2.)
        else:
            psih  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - (1. + (2./3.) * zeta) ** (1.5) - (10./3.) / 0.35 + 1.
        return psih
 
    def jarvis_stewart(self):
        # calculate surface resistances using Jarvis-Stewart model
        if(self.sw_rad):
            f1 = 1. / min(1.,((0.004 * self.Swin + 0.05) / (0.81 * (0.004 * self.Swin + 1.))))
        else:
            f1 = 1.
  
        if(self.w2 > self.wwilt):# and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.e8
 
        # Limit f2 in case w2 > wfc, where f2 < 1
        #f2 = max(f2, 1.);
 
        f3 = 1. / np.exp(- self.gD * (self.esat - self.e2m) / 100.)
        f4 = 1./ (1. - 0.0016 * (298.0-self.T2m)**2.)
  
        self.rs = self.rsmin / self.LAI * f1 * f2 * f3 * f4
        #self.glw = 1/self.rsmin /(f1 * f2 * f3 * f4)       # I do not know how to downscale to leaf level. This is not correct
        #self.glw = 1/self.rs * self.Kx[0]/(1-np.exp(-self.Kx[0]*self.LAI))
    def factorial(self,k):
        factorial = 1
        for n in range(2,k+1):
            factorial = factorial * float(n)
        return factorial;

    def E1(self,x):
        E1sum = 0
        for k in range(1,100):
            E1sum += (-x)**k/k/self.factorial(k)
            #E1sum += pow((-1.),(k + 0.0)) * pow(x,(k + 0.0)) / ((k + 0.0) * self.factorial(k))
        return -0.57721566490153286060 - np.log(x) - E1sum
 
    def ags(self):
        # Select index for plant type
        if(self.c3c4 == 'c3'):
            c = 0
        elif(self.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('Invalid option \"%s\" for \"c3c4\"'%self.c3c4)

        # calculate CO2 compensation concentration
        T = self.thetasurf #
        T = self.T2m
        T = self.T0_105m
        e = self.e2m
        e = self.e0_105m
        #e = self.qsurf/(0.622)*self.Ps
        #e = self.e
        CO2comp       = self.CO2comp298[c] * self.rho * pow(self.Q10CO2[c],(0.1 * (T - 298.)))  
        self.CO2comp  = CO2comp
        # calculate mesophyll conductance
        gm            = self.gm298[c] *  pow(self.Q10gm[c],(0.1 * (T-298.))) \
                          / ( (1. + np.exp(0.3 * (self.T1gm[c] - T))) \
                             * (1. + np.exp(0.3 * (T - self.T2gm[c]))))
        gm            = gm / 1000. # conversion from mm s-1 to m s-1
        self.gm       = gm
        # calculate CO2 concentration inside the leaf (ci)
        fmin0         = self.gmin[c]/ self.nuco2q - 1. / 9. * gm
        self.fmin0    = fmin0
        fmin          = (-fmin0 + pow((pow(fmin0,2.) + 4 * self.gmin[c]/self.nuco2q * gm),0.5)) / (2. * gm)
        self.fmin     = fmin
        #Ds            = (esat(self.Ts) - self.e) / 1000. # kPa
        Ds            = (esat(T)-e)/1000
        self.Ds       = Ds * 1000  # [Pa]
        D0            = (self.f0[c] - fmin) / self.ad[c]
        self.D0       = D0 * 1000           # Added by Raquel in Pa
        a1            = 1. / (1. - self.f0[c])
        self.a1       = a1
        Dstar         = D0 / (a1 * (self.f0[c] - fmin))
        self.Dstar    = Dstar*1000 # (Pa)
        cfrac         = self.f0[c] * (1. - (Ds / D0)) + fmin * (Ds / D0)
        self.cfrac    = cfrac
        co2abs        = self.CO2 * (self.mco2 / self.mair) * self.rho # conversion mumol mol-1 (ppm) to mgCO2 m3
        self.co2abs   = co2abs
        ci            = cfrac * (co2abs - CO2comp) + CO2comp
        self.ci       = ci
  
        # calculate maximal gross primary production in high light conditions (Ag)
        Ammax         = self.Ammax298[c] *  pow(self.Q10Am[c],(0.1 * (T - 298.))) / ( (1. + np.exp(0.3 * (self.T1Am[c] - T))) * (1. + np.exp(0.3 * (T - self.T2Am[c]))))
        self.Ammax    = Ammax

        # calculate gross assimilation rate (Am)
        Am           = Ammax * (1. - np.exp(-(gm * (ci - CO2comp) / Ammax)))
        self.Am      = Am
        Rdark        = (1. / 9.) * Am
        self.Rdark   = Rdark
        #PARt         = 0.465 * max(1e-1,self.Swin) 
        PARt         = 0.465*self.Swin
        PAR          = self.Kx[c]*PARt
        self.PAR = PAR
        PAR2_3       = self.Kx[c]*PARt*np.exp(-self.Kx[c]*1/3)
        PAR1_3       = self.Kx[c]*PARt*np.exp(-self.Kx[c]*2/3)
        PAR0         = self.Kx[c]*PARt*np.exp(-self.Kx[c])
        # calculate  light use efficiency
        alphac       = self.alpha0[c] * (co2abs - CO2comp) / (co2abs + 2. * CO2comp)
        self.alphac  = alphac
        # calculate gross primary productivity without any soil water stress
        Agstar       = (Am + Rdark) * (1 - np.exp((-alphac * PAR)/ (Am + Rdark)))
        self.Agstar = Agstar
        # calculate effect of soil moisture stress on gross assimilation rate
        betaw         = max(1e-3, min(1.,(self.w2 - self.wwilt)/(self.wfc - self.wwilt)))
  
        # calculate stress function
        if (self.c_beta == 0):
            fstr = betaw;
        else:
            # Following Combe et al (2016)
            if (self.c_beta < 0.25):
                PCbeta = 6.4 * self.c_beta
            elif (self.c_beta < 0.50):
                PCbeta = 7.6 * self.c_beta - 0.3
            else:
                PCbeta = 2**(3.66 * self.c_beta + 0.34) - 1
            fstr = (1. - np.exp(-PCbeta * betaw)) / (1 - np.exp(-PCbeta))
            self.PCbeta  = PCbeta
        
        #fstr = 1 # addition to impose no moisture restriction
        
        self.fstr    = fstr
        
        # calculate gross primary productivity
        Ag           = (Am + Rdark) * (1 - np.exp(-alphac * PAR / (Am + Rdark)))*fstr
        Ag2_3        = (Am + Rdark) * (1 - np.exp(-alphac * PAR2_3 / (Am + Rdark)))*fstr
        Ag1_3        = (Am + Rdark) * (1 - np.exp(-alphac * PAR1_3 / (Am + Rdark)))*fstr
        Ag0          = (Am + Rdark) * (1 - np.exp(-alphac * PAR0 / (Am + Rdark)))*fstr
        self.Ag      = Ag
        
        Aln      = Ag - Rdark
        Aln2_3   = Ag2_3 - Rdark
        Aln1_3   = Ag1_3 - Rdark
        Aln0     = Ag0 - Rdark
        
        
        # calculate stomata conductance at leaf level
        glc        = self.gmin[c]/self.nuco2q + Aln * a1 / (co2abs - CO2comp) / (1 + Ds/Dstar)
        glc2_3     = self.gmin[c]/self.nuco2q + Aln2_3 * a1 / (co2abs - CO2comp) / (1 + Ds/Dstar)
        glc1_3        = self.gmin[c]/self.nuco2q + Aln1_3 * a1 / (co2abs - CO2comp) / (1 + Ds/Dstar)
        glc0        = self.gmin[c]/self.nuco2q + Aln0 * a1 / (co2abs - CO2comp) / (1 + Ds/Dstar)
        
        
        
        if (self.sw_gsdelay) and (self.t != 0): # Stomatal delay scheme based on (M. Sikma et.al., 2018) (P. J. Sellers et.al., 1996 (appendix C))
            glc_inf = glc
            glc = self.glc*np.exp(-self.dt/self.taug)+(1-np.exp(-self.dt/self.taug))*glc_inf
        
        glw        = glc * self.nuco2q
        LEleaf     = self.rho * self.Lv * glw * (qsat(T, self.Ps) - self.q0_105m)
        
        self.glc      = glc
        self.glc2_3   = glc2_3
        self.glc1_3   = glc1_3
        self.glc0     = glc0
        self.glw      = glw
        self.LEleaf   = LEleaf
        self.Aln      = Ag - Rdark
        
        ## Canopy level 
        
        # 1.- calculate upscaling from leaf to canopy: net flow CO2 into the plant (An)
        y            =  alphac * self.Kx[c] * PARt / (Am + Rdark)
        Ag1          = ((Am + Rdark) * (1 - 1. / (self.Kx[c])/self.LAI * (self.E1(y * np.exp(-self.Kx[c] * self.LAI)) - self.E1(y))))*fstr
        An1          = Ag1-Rdark*self.LAI*fstr
        
        self.Ag1     = Ag1
    
        # 2.- calculate upscaling from leaf to canopy: CO2 conductance at canopy level
        
  
        gcco2        = self.LAI * (self.gmin[c]/ self.nuco2q + a1 * An1 / ((co2abs - CO2comp) * (1. + Ds / Dstar)))
  
        # calculate surface resistance for moisture and carbon dioxide
        self.rs      = 1. / (1.6 * gcco2)
        rsCO2        = 1. / gcco2
  
        # calculate net flux of CO2 into the plant (An)
        An           = -(co2abs - ci) / (self.ra + rsCO2)
        #An            = - Ag1*rsCO2/(self.ra + rsCO2)
        self.An      = An
    
        # CO2 soil surface flux
        fw           = self.Cw * self.wmax / (self.wg + self.wmin)
        Resp         = self.R10 * (1. - fw) * np.exp(self.E0 / (283.15 * 8.314) * (1. - 283.15 / (self.Tsoil)))
        self.Resp    = Resp
    
        # CO2 flux
        self.wCO2A   = An   * (self.mair / (self.rho * self.mco2))
        self.wCO2R   = Resp * (self.mair / (self.rho * self.mco2))
        self.wCO2    = (self.wCO2A + self.wCO2R) * self.cveg
        
        
    def FvCB_M(self):
        ## Farquhar-Medlyn implementation without plant water stress functions
        # leaf level
        def FvCB_M_leaf(self, a):
            # Calculation of needed parameters
            #Jmax298 = 2*self.Vcmax298             # maximum electron transport rate at 298 K (mol electrons /m2/s)
            Jmax298 = self.Jmax298                # maximum electron transport rate at 298 K (mol electrons /m2/s)
            Rd298 = 0.025*self.Vcmax298#0.015*self.Vcmax298           # dark respiration at 298 K (mol CO2/m2/s)
    
            # Implementation of T dependency of intermediate variables with a Q10 temperature response function
            Tleaf = self.thetasurf               # Underlying assumption: temperature of the leaf is the surface temperature
            Vcmax = self.Vcmax298*2.21**(0.1*(Tleaf-298))
            Jmax = Jmax298*1.65**(0.1*(Tleaf-298))
            Kc = self.Kc298*2.24**(0.1*(Tleaf-298))
            Ko = self.Ko298*1.63**(0.1*(Tleaf-298))
            CO2comp = self.CO2comp298_*1.37**(0.1*(Tleaf-298))
            Rd = Rd298*2.46**(0.1*(Tleaf-298))
            
            # Conversion of PAR (W/m2) at the photon flux density, Q (mol photons/m2/s)
            Na         = 6.0221408e23        # Avogadro's number (# of particles in 1 mol)
            h          = 6.62607015e-34      # Planck's constant (J*s)
            c          = 299792458           # Light speed constant (m/s)
            lambda_PAR = 550e-9              # Wavelenght of mean photon in the active photosynthetic range (m)
    
            from_PAR_to_Q = 1/(Na*h*c/lambda_PAR) 
            PAR = 0.5*max(1e-1, self.Swin)
            PARleaf = self.Kx_ * PAR * np.exp(-a*self.Kx_)  # (W/m2leaf)
            # Calculation of photon flux density 
            Q = PARleaf*from_PAR_to_Q       # (mol photons/m2/s)
            
            # Calculus of leaf to air vapor pressure deficit (kPa)
            Ds            = (esat(Tleaf)-self.e)/1000
            
            # Calculus of the potential rate of electrons, J (mol e-/m2/s)
            abs_ = 0.85 # no idea (-)
            Theta_ = 0.7 # no idea (-)
            J2 = 0.5* Q*(1-self.f_)*abs_
            J = (J2+Jmax-np.sqrt((J2+Jmax)**2-4*Theta_*J2*Jmax))/(2*Theta_) # (mol e-/m2/s)
            
            # calculate effect of soil moisture stress on stomatal conductance
            betaw         = max(1e-3, min(1.,(self.w2 - self.wwilt)/(self.wfc - self.wwilt)))
  
            # calculate stress function
            if (self.c_beta == 0):
                fstr = betaw;
            else:
                # Following Combe et al (2016)
                if (self.c_beta < 0.25):
                    P = 6.4 * self.c_beta
                elif (self.c_beta < 0.50):
                    P = 7.6 * self.c_beta - 0.3
                else:
                    P = 2**(3.66 * self.c_beta + 0.34) - 1
            fstr = (1. - np.exp(-P * betaw)) / (1 - np.exp(-P))
        
            fstr = 1 # addition to impose no moisture restriction
            self.fstr = fstr
        
            # Initialize Ci as Ci = 0.7 Ca
            Ci_ppm = 0.7*self.CO2                                        # (ppm = micromol CO2/mol air)
            Ci = Ci_ppm*1e-6                                             # (mol CO2/mol air)
            CO2 = self.CO2*1e-6                                          # (mol CO2/ mol air)
        
            for i in np.arange(10):
                # Calculus of Rubisco-limited rate of CO2 assimilation (mol CO2 m-2 s-1)
                Wc = Vcmax*Ci/(Ci+Kc*(1+self.Oi/Ko))
                
                # Calculus of electron transport-limited rate of CO2 assimilation (mol CO2 m-2 s-1)
                Wj = J/(4+8*CO2comp/Ci)
        
                # Calculus of gross photosynthesis rate (mol CO2 m-2 s-1)
                Vc = min(Wc,Wj)
                Ag = Vc*(1-CO2comp/Ci)
                # Calculus of net assimilation rate (mol CO2 m-2 s-1)
                An = Ag - Rd                                             # (mol CO2/m2leaf/s )
        
                # Calculus of stomatal conductance according to Medlyn et al. (2011)
                glc = self.g0_ + An*fstr/CO2*(1+self.g1_/np.sqrt(Ds))         # (mol air/m2leaf/s)
        
                Ci = CO2 - An/glc
            return Ag, An, glc, Ci, Rd
        
        # Save values at leaf level
        Ag, An, glc, Ci, Rd = FvCB_M_leaf(self,a = 0)
        Ag_mg_m2_s = Ag * self.mco2*1e3     # conversion from mol CO2/m2leaf/s to mg CO2/m2 leaf /s
        self.Ag = Ag_mg_m2_s
        Rd_mg_m2_s = Rd * self.mco2*1e3
        self.Rdark = Rd_mg_m2_s
        glc_m_s = glc*0.0248                # conversion from mol air/m2leaf/s to m/s
        self.glc = glc_m_s
        self.glw = glc_m_s*self.nuco2q
        
        # Canopy level
        canopy_levels = 50
        a_ = np.linspace(0, self.LAI, canopy_levels)
        
        An_levels = []
        glc_levels = []
        Ci_levels = []
        An_levels_wind = []
        
        for a_i in a_:
            Ag_i, An_i, glc_i, Ci_i, Rd_i = FvCB_M_leaf(self, a = a_i)
            An_levels.append(An_i)
            glc_levels.append(glc_i)
            Ci_levels.append(Ci_i*1e6)
            An_levels_wind.append(-(Ci_i-self.CO2*1e-6)/(1/(glc_i*0.0248)+self.ra)*self.mco2/self.mair*self.rho*1e6)    # mg CO2/m2/s
            
        
        An_levels = np.array(An_levels)
        glc_levels = np.array(glc_levels)
        An_levels_wind = np.array(An_levels_wind)
        Ci_levels = np.array(Ci_levels)
        
        An_canopy = integration.area_Riemman(An_levels, self.LAI/canopy_levels)   # (mol CO2/m2leaf/s )
        An_canopy_mg_m2_s = An_canopy * self.mco2*1e3     # conversion from mol CO2/m2leaf/s to mg CO2/m2/s
        
        An_wind_canopy_mg_m2_s = integration.area_Riemman(An_levels_wind, self.LAI/canopy_levels)

        gcco2_canopy = integration.area_Riemman(glc_levels, self.LAI/canopy_levels)   # (mol air/m2leaf/s )
        gcco2_canopy_m_s = gcco2_canopy*0.0248     # conversion from mol air/m2leaf/s to m/s
        
        
        rcco2 = 1/gcco2_canopy_m_s
        # Way to do it but we need to convert units
        #An_canopy = -(self.CO2-Ci_levels)/(self.ra + rcco2)
        # No aerodynamic resistance included yet and not soil moisture stress response yet
        self.rs = 1/(1.6*gcco2_canopy_m_s)
        
        # CO2 soil surface flux
        fw           = self.Cw * self.wmax / (self.wg + self.wmin)
        Resp         = self.R10 * (1. - fw) * np.exp(self.E0 / (283.15 * 8.314) * (1. - 283.15 / (self.Tsoil)))
        self.Resp    = Resp
    
    
        # CO2 flux
        self.wCO2A   = -An_wind_canopy_mg_m2_s   * (self.mair / (self.rho * self.mco2))

        self.wCO2R   = Resp * (self.mair / (self.rho * self.mco2))
        self.wCO2    = (self.wCO2A + self.wCO2R) * self.cveg
        
    def run_land_surface(self):
        # compute ra
        ueff = np.sqrt(self.u ** 2. + self.v ** 2. + self.wstar**2.)

        if(self.sw_sl):
          self.ra = (self.Cs * ueff)**-1.
        else:
          self.ra = ueff / max(1.e-3, self.ustar)**2.

        # first calculate essential thermodynamic variables
        self.esat    = esat(self.theta)
        self.qsat    = qsat(self.theta, self.Ps)
        desatdT      = self.esat * (17.2694 / (self.theta - 35.86) - 17.2694 * (self.theta - 273.16) / (self.theta - 35.86)**2.)
        self.dqsatdT = 0.622 * desatdT / self.Ps
        self.e       = self.q * self.Ps / 0.622

        if(self.ls_type == 'js'): 
            self.jarvis_stewart() 
        elif(self.ls_type == 'ags'):
            self.ags()
        elif(self.ls_type == 'FvCB_M'):
            self.FvCB_M()
        else:
            sys.exit('option \"%s\" for \"ls_type\" invalid'%self.ls_type)

        # recompute f2 using wg instead of w2
        if(self.wg > self.wwilt):# and self.w2 <= self.wfc):
          f2          = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
          f2        = 1.e8
        self.rssoil = self.rssoilmin * f2 
 
        Wlmx = self.LAI * self.Wmax
        self.cliq = min(1., self.Wl / Wlmx) 

     
        # calculate skin temperature implictly
        #self.Ts   = (self.Q  + self.rho * self.cp / self.ra * self.theta \
         #   + self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs    ) * (self.dqsatdT * self.theta - self.qsat + self.q) \
         #   + (1. - self.cveg)             * self.rho * self.Lv / (self.ra + self.rssoil) * (self.dqsatdT * self.theta - self.qsat + self.q) \
         #   + self.cveg * self.cliq        * self.rho * self.Lv /  self.ra                * (self.dqsatdT * self.theta - self.qsat + self.q) + self.Lambda * self.Tsoil) \
         #   / (self.rho * self.cp / self.ra + self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs) * self.dqsatdT \
         #   + (1. - self.cveg) * self.rho * self.Lv / (self.ra + self.rssoil) * self.dqsatdT + self.cveg * self.cliq * self.rho * self.Lv / self.ra * self.dqsatdT + self.Lambda)

        ## Raquel's addition to reduce number of operations of self.Ts
        self.gveg_eff = ((1. - self.cliq)*self.cveg)/(self.ra + self.rs)
        self.gliq_eff = (self.cliq * self.cveg) / self.ra
        self.gsoil_eff = (1. - self.cveg)/ (self.ra + self.rssoil)
        self.R = 1 / (self.gveg_eff + self.gsoil_eff + self.gliq_eff)
        geff = self.gveg_eff + self.gliq_eff + self.gsoil_eff
        
        # self.Ts = (self.Q + self.rho * self.cp / self.ra * self.theta + (self.dqsatdT * self.theta - self.qsat + self.q) * self.rho * self.Lv \
        #     * (self.cveg * (1 - self.cliq) / (self.ra + self.rs) + (1 - self.cveg) / (self.ra + self.rssoil) + self.cveg * self.cliq / self.ra) + self.Lambda * self.Tsoil) \
        #     / (self.rho * self.cp / self.ra + self.dqsatdT * self.rho * self.Lv * (self.cveg * (1 - self.cliq) / (self.ra + self.rs) + (1 - self.cveg) / (self.ra + self.rssoil) \
        #     + self.cveg * self.cliq / self.ra) + self.Lambda)
        
        self.Ts = (self.Q + (self.cp/self.ra + self.Lv/self.R*self.dqsatdT)*\
                   self.rho*self.theta-self.rho*self.Lv/self.R*(self.qsat-self.q)\
                       +self.Lambda*self.Tsoil)/(self.rho*self.cp/self.ra + \
                        self.rho*self.Lv/self.R*self.dqsatdT+self.Lambda)
        
        ## Definition of Ts according to 
        
        #self.G      = self.Q/5
        #self.Ts     = (self.Q-self.G-self.rho*self.Lv/self.R*(self.qsat-self.q))/(self.rho*self.cp/self.ra+self.rho*self.Lv/self.R*self.dqsatdT)+self.theta
        #self.Tsoil  = self.Ts - self.G/self.Lambda
        
        esatsurf      = esat(self.Ts)
        self.qsatsurf = qsat(self.Ts, self.Ps)

        self.ETcommonpart = self.rho * self.Lv * (self.dqsatdT * (self.Ts - self.theta) + self.qsat - self.q)
        
        self.LEveg = self.ETcommonpart * self.gveg_eff         
        self.LEliq = self.ETcommonpart * self.gliq_eff        
        self.LEsoil = self.ETcommonpart * self.gsoil_eff
        
        """
            Raquel's addition of a total resistance part and parts of LE
        """
        
        self.LE_radiation = self.rho * self.Lv * (self.dqsatdT * (self.Ts - self.theta)) / self.R
        self.LE_VPD = self.rho * self.Lv * (self.qsat - self.q) / self.R
        self.LE_ideal = self.rho * self.Lv * (self.qsatsurf - self.q) / self.R
        
        
        self.Wltend      = - self.LEliq / (self.rhow * self.Lv)
  
        self.LE     = self.LEsoil + self.LEveg + self.LEliq
        self.H      = self.rho * self.cp / self.ra * (self.Ts - self.theta)
        self.G      = self.Lambda * (self.Ts - self.Tsoil)
        self.LEpot  = (self.dqsatdT * (self.Q - self.G) + self.rho * self.cp / self.ra * (self.qsat - self.q)) / (self.dqsatdT + self.cp / self.Lv)
        self.LEref  = (self.dqsatdT * (self.Q - self.G) + self.rho * self.cp / self.ra * (self.qsat - self.q)) / (self.dqsatdT + self.cp / self.Lv * (1. + self.rsmin / self.LAI / self.ra))
        
        CG          = self.CGsat * (self.wsat / self.w2)**(self.b / (2. * np.log(10.)))
  
        self.Tsoiltend   = CG * self.G - 2. * np.pi / 86400. * (self.Tsoil - self.T2)
   
        d1          = 0.1
        C1          = self.C1sat * (self.wsat / self.wg) ** (self.b / 2. + 1.)
        C2          = self.C2ref * (self.w2 / (self.wsat - self.w2) )
        wgeq        = self.w2 - self.wsat * self.a * ( (self.w2 / self.wsat) ** self.p * (1. - (self.w2 / self.wsat) ** (8. * self.p)) )
        self.wgtend = - C1 / (self.rhow * d1) * self.LEsoil / self.Lv - C2 / 86400. * (self.wg - wgeq)
  
        # calculate kinematic heat fluxes
        self.wtheta   = self.H  / (self.rho * self.cp)
        self.wq       = self.LE / (self.rho * self.Lv)
 
    def integrate_land_surface(self):
        # integrate soil equations
        Tsoil0        = self.Tsoil
        wg0           = self.wg
        Wl0           = self.Wl
  
        self.Tsoil    = Tsoil0  + self.dt * self.Tsoiltend
        self.wg       = wg0     + self.dt * self.wgtend
        self.Wl       = Wl0     + self.dt * self.Wltend
  
    # store model output
    def store(self):
        t                      = self.t
        self.out.t[t]          = t * self.dt / 3600. + self.tstart
        self.out.h[t]          = self.h
        
        self.out.theta[t]      = self.theta
        self.out.thetav[t]     = self.thetav
        self.out.dtheta[t]     = self.dtheta
        self.out.dthetav[t]    = self.dthetav
        self.out.wtheta[t]     = self.wtheta
        self.out.wthetav[t]    = self.wthetav
        self.out.wthetae[t]    = self.wthetae
        self.out.wthetave[t]   = self.wthetave
        self.out.advtheta[t]   = self.advtheta
        self.out.gammatheta[t] = self.gammatheta
        
        self.out.q[t]          = self.q
        self.out.dq[t]         = self.dq
        self.out.wq[t]         = self.wq
        self.out.wqe[t]        = self.wqe
        self.out.wqM[t]        = self.wqM
        self.out.advq[t]       = self.advq
        self.out.gammaq[t]     = self.gammaq
      
        self.out.qsat[t]       = self.qsat
        self.out.e[t]          = self.e
        self.out.esat[t]       = self.esat
      
        fac = (self.rho*self.mco2)/self.mair
        self.out.CO2[t]        = self.CO2
        self.out.dCO2[t]       = self.dCO2
        self.out.wCO2[t]       = self.wCO2  * fac
        self.out.wCO2e[t]      = self.wCO2e * fac
        self.out.wCO2R[t]      = self.wCO2R * fac
        self.out.wCO2A[t]      = self.wCO2A * fac

        self.out.u[t]          = self.u
        self.out.du[t]         = self.du
        self.out.uw[t]         = self.uw
        
        self.out.v[t]          = self.v
        self.out.dv[t]         = self.dv
        self.out.vw[t]         = self.vw
        self.out.we[t]         = self.we
        
        self.out.T2m[t]        = self.T2m
        self.out.q2m[t]        = self.q2m
        self.out.u2m[t]        = self.u2m
        self.out.v2m[t]        = self.v2m
        self.out.e2m[t]        = self.e2m
        self.out.esat2m[t]     = self.esat2m
        
        self.out.T0_105m[t]    = self.T0_105m
        self.out.q0_105m[t]    = self.q0_105m
        self.out.u0_105m[t]    = self.u0_105m
        self.out.v0_105m[t]    = self.v0_105m
        self.out.e0_105m[t]    = self.e0_105m
        self.out.esat0_105m[t] = self.esat0_105m
        
        self.out.thetasurf[t]  = self.thetasurf
        self.out.thetavsurf[t] = self.thetavsurf
        self.out.qsurf[t]      = self.qsurf
        self.out.ustar[t]      = self.ustar
        self.out.Cm[t]         = self.Cm
        self.out.Cs[t]         = self.Cs
        self.out.L[t]          = self.L
        self.out.Rib[t]        = self.Rib
  
        self.out.Swin[t]       = self.Swin
        self.out.Swout[t]      = self.Swout
        self.out.Lwin[t]       = self.Lwin
        self.out.Lwout[t]      = self.Lwout
        self.out.Q[t]          = self.Q
  
        self.out.ra[t]         = self.ra
        self.out.rs[t]         = self.rs
        self.out.H[t]          = self.H
        self.out.LE[t]         = self.LE
        self.out.LEliq[t]      = self.LEliq
        self.out.LEveg[t]      = self.LEveg
        self.out.LEsoil[t]     = self.LEsoil
        self.out.LEpot[t]      = self.LEpot
        self.out.LEref[t]      = self.LEref
        self.out.G[t]          = self.G
        
        # added by Raquel
        self.out.Tsoil[t]        = self.Tsoil        # temperature top soil layer [K]
        self.out.ETcommonpart[t] = self.ETcommonpart
        self.out.LE_radiation[t] = self.LE_radiation
        self.out.LE_VPD[t]       = self.LE_VPD
        self.out.LE_ideal[t]     = self.LE_ideal
        self.out.R[t]            = self.R
        self.out.gveg_eff[t]     = self.gveg_eff
        self.out.gsoil_eff[t]    = self.gsoil_eff
        self.out.gliq_eff[t]     = self.gliq_eff
        self.out.rssoil[t]       = self.rssoil
        self.out.Ts[t]           = self.Ts
        self.out.dqsatdT[t]      = self.dqsatdT
        self.out.D0[t]           = self.D0
        self.out.Ds[t]           = self.Ds
        self.out.PAR[t]          = self.PAR
        # soil
        self.out.wg[t]         = self.wg
        self.out.w2[t]         = self.w2
        ## A-gs
        self.out.co2abs[t]     = self.co2abs
        self.out.ci[t]         = self.ci
        self.out.Ammax[t]      = self.Ammax
        self.out.Am[t]         = self.Am
        self.out.Rdark[t]      = self.Rdark
        self.out.Ag[t]         = self.Ag
        self.out.Agstar[t]     = self.Agstar
        self.out.Ag1[t]        = self.Ag1
        self.out.An[t]         = self.An
        self.out.Resp[t]       = self.Resp
        self.out.fstr[t]       = self.fstr
        self.out.PCbeta[t]     = self.PCbeta
        self.out.LEleaf[t]     = self.LEleaf  
        self.out.glw[t]        = self.glw  
        self.out.glc[t]        = self.glc 
        self.out.a1[t]         = self.a1
        self.out.alphac[t]     = self.alphac
        self.out.CO2comp[t]    = self.CO2comp
        self.out.Dstar[t]      = self.Dstar
        self.out.fmin[t]       = self.fmin
        self.out.fmin0[t]      = self.fmin0
        self.out.gm[t]         = self.gm
        self.out.cfrac[t]      = self.cfrac
        self.out.Aln[t]        = self.Aln
        self.out.glc2_3[t]     = self.glc2_3 
        self.out.glc1_3[t]     = self.glc1_3
        self.out.glc0[t]       = self.glc0
        self.out.wstar[t]      = self.wstar
        # end of Raquel's additions #
        
        
        self.out.zlcl[t]       = self.lcl
        self.out.RH_h[t]       = self.RH_h

        self.out.ac[t]         = self.ac
        self.out.M[t]          = self.M
        self.out.dz[t]         = self.dz_h
  
    # delete class variables to facilitate analysis in ipython
    def exitmodel(self):
        del(self.Lv)
        del(self.cp)
        #del(self.rho)
        del(self.k)
        del(self.g)
        del(self.Rd)
        del(self.Rv)
        del(self.bolz)
        del(self.S0)
        del(self.rhow)
  
        del(self.t)
        del(self.dt)
        del(self.tsteps)
         
        del(self.h)          
        del(self.Ps)        
        del(self.fc)        
        del(self.ws)
        
        del(self.theta)
        del(self.dtheta)
        del(self.gammatheta)
        del(self.advtheta)
        del(self.beta)
        del(self.wtheta)
    
        del(self.T2m)
        del(self.q2m)
        del(self.e2m)
        del(self.esat2m)
        del(self.u2m)
        del(self.v2m)
        
        #del(self.thetasurf)
        #del(self.qsatsurf)
        del(self.thetav)
        del(self.dthetav)
        del(self.thetavsurf)
        #del(self.qsurf)
        del(self.wthetav)
        
        del(self.q)
        del(self.qsat)
        #del(self.dqsatdT)
        del(self.e)
        del(self.esat)
        del(self.dq)
        del(self.gammaq)
        #del(self.advq)
        del(self.wq)
        
        del(self.u)
        del(self.du)
        del(self.gammau)
        del(self.advu)
        
        del(self.v)
        del(self.dv)
        del(self.gammav)
        del(self.advv)
  
        del(self.htend)
        del(self.thetatend)
        del(self.dthetatend)
        del(self.qtend)
        del(self.dqtend)
        del(self.utend)
        del(self.dutend)
        del(self.vtend)
        del(self.dvtend)
     
        del(self.Tsoiltend) 
        del(self.wgtend)  
        del(self.Wltend) 
  
        del(self.ustar)
        del(self.uw)
        del(self.vw)
        del(self.z0m)
        del(self.z0h)        
        del(self.Cm)         
        del(self.Cs)
        del(self.L)
        del(self.Rib)
        del(self.ra)
  
        del(self.lat)
        del(self.lon)
        del(self.doy)
        del(self.tstart)
   
        del(self.Swin)
        del(self.Swout)
        del(self.Lwin)
        del(self.Lwout)
        del(self.cc)
  
 #       del(self.wg)
 #       del(self.w2)
        del(self.cveg)
        del(self.cliq)
        del(self.Tsoil)
        del(self.T2)
        del(self.a)
        del(self.b)
        del(self.p)
        del(self.CGsat)
  
        del(self.wsat)
        del(self.wfc)
        del(self.wwilt)
  
        del(self.C1sat)
        del(self.C2ref)
  
        del(self.LAI)
        del(self.rs)
        #del(self.rssoil)
        del(self.rsmin)
        del(self.rssoilmin)
        del(self.alpha)
        del(self.gD)
  
        # del(self.Ts)
  
        del(self.Wmax)
        del(self.Wl)
  
        del(self.Lambda)
        
        del(self.Q)
        del(self.H)
        del(self.LE)
        del(self.LEliq)
        del(self.LEveg)
        del(self.LEsoil)
        del(self.LEpot)
        del(self.LEref)
        del(self.G)
  
        del(self.sw_ls)
        del(self.sw_rad)
        del(self.sw_sl)
        del(self.sw_wind)
        del(self.sw_shearwe)

# class for storing mixed-layer model output data
class model_output:
    def __init__(self, tsteps):
        self.t          = np.zeros(tsteps)    # time [s]

        # mixed-layer variables
        self.h          = np.zeros(tsteps)    # ABL height [m]
      
        self.theta      = np.zeros(tsteps)    # initial mixed-layer potential temperature [K]
        self.thetav     = np.zeros(tsteps)    # initial mixed-layer virtual potential temperature [K]
        self.dtheta     = np.zeros(tsteps)    # initial potential temperature jump at h [K]
        self.dthetav    = np.zeros(tsteps)    # initial virtual potential temperature jump at h [K]
        self.wtheta     = np.zeros(tsteps)    # surface kinematic heat flux [K m s-1]
        self.wthetav    = np.zeros(tsteps)    # surface kinematic virtual heat flux [K m s-1]
        self.wthetae    = np.zeros(tsteps)    # entrainment kinematic heat flux [K m s-1]
        self.wthetave   = np.zeros(tsteps)    # entrainment kinematic virtual heat flux [K m s-1]
        self.advtheta   = np.zeros(tsteps)    # advection of potential temperature [K s-1]
        self.gammatheta = np.zeros(tsteps)    # free atmosphere potential temperature lapse rate [K m-1]
        
        self.q          = np.zeros(tsteps)    # mixed-layer specific humidity [kg kg-1]
        self.dq         = np.zeros(tsteps)    # initial specific humidity jump at h [kg kg-1]
        self.wq         = np.zeros(tsteps)    # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wqe        = np.zeros(tsteps)    # entrainment kinematic moisture flux [kg kg-1 m s-1]
        self.wqM        = np.zeros(tsteps)    # cumulus mass-flux kinematic moisture flux [kg kg-1 m s-1]
        self.we         = np.zeros(tsteps)    #entrainment velocity [m s-1]
        self.advq       = np.zeros(tsteps)    # advection of moisture [kg water kg-1 air s-1]
        self.gammaq     = np.zeros(tsteps)    # free atmosphere specific humidity lapse rate [kg kg-1 m-1]

        self.qsat       = np.zeros(tsteps)    # mixed-layer saturated specific humidity [kg kg-1]
        self.e          = np.zeros(tsteps)    # mixed-layer vapor pressure [Pa]
        self.esat       = np.zeros(tsteps)    # mixed-layer saturated vapor pressure [Pa]

        self.CO2        = np.zeros(tsteps)    # mixed-layer CO2 [ppm]
        self.dCO2       = np.zeros(tsteps)    # initial CO2 jump at h [ppm]
        self.wCO2       = np.zeros(tsteps)    # surface total CO2 flux [mgC m-2 s-1]
        self.wCO2A      = np.zeros(tsteps)    # surface assimilation CO2 flux [mgC m-2 s-1]
        self.wCO2R      = np.zeros(tsteps)    # surface respiration CO2 flux [mgC m-2 s-1]
        self.wCO2e      = np.zeros(tsteps)    # entrainment CO2 flux [mgC m-2 s-1]
        self.wCO2M      = np.zeros(tsteps)    # CO2 mass flux [mgC m-2 s-1]
        
        self.u          = np.zeros(tsteps)    # initial mixed-layer u-wind speed [m s-1]
        self.du         = np.zeros(tsteps)    # initial u-wind jump at h [m s-1]
        self.uw         = np.zeros(tsteps)    # surface momentum flux u [m2 s-2]
        
        self.v          = np.zeros(tsteps)    # initial mixed-layer u-wind speed [m s-1]
        self.dv         = np.zeros(tsteps)    # initial u-wind jump at h [m s-1]
        self.vw         = np.zeros(tsteps)    # surface momentum flux v [m2 s-2]

        # diagnostic meteorological variables
        self.T2m        = np.zeros(tsteps)    # 2m temperature [K]   
        self.q2m        = np.zeros(tsteps)    # 2m specific humidity [kg kg-1]
        self.u2m        = np.zeros(tsteps)    # 2m u-wind [m s-1]    
        self.v2m        = np.zeros(tsteps)    # 2m v-wind [m s-1]    
        self.e2m        = np.zeros(tsteps)    # 2m vapor pressure [Pa]
        self.esat2m     = np.zeros(tsteps)    # 2m saturated vapor pressure [Pa]
        
        self.T0_105m    = np.zeros(tsteps)    # 0.105m temperature [K]   
        self.q0_105m    = np.zeros(tsteps)    # 0.105m specific humidity [kg kg-1]
        self.u0_105m    = np.zeros(tsteps)    # 0.105m u-wind [m s-1]    
        self.v0_105m    = np.zeros(tsteps)    # 0.105m v-wind [m s-1]    
        self.e0_105m    = np.zeros(tsteps)    # 0.105m vapor pressure [Pa]
        self.esat0_105m = np.zeros(tsteps)    # 0.105m saturated vapor pressure [Pa]

        # surface-layer variables
        self.thetasurf  = np.zeros(tsteps)    # surface potential temperature [K]
        self.thetavsurf = np.zeros(tsteps)    # surface virtual potential temperature [K]
        self.qsurf      = np.zeros(tsteps)    # surface specific humidity [kg kg-1]
        self.ustar      = np.zeros(tsteps)    # surface friction velocity [m s-1]
        self.z0m        = np.zeros(tsteps)    # roughness length for momentum [m]
        self.z0h        = np.zeros(tsteps)    # roughness length for scalars [m]
        self.Cm         = np.zeros(tsteps)    # drag coefficient for momentum []
        self.Cs         = np.zeros(tsteps)    # drag coefficient for scalars []
        self.L          = np.zeros(tsteps)    # Obukhov length [m]
        self.Rib        = np.zeros(tsteps)    # bulk Richardson number [-]

        # radiation variables
        self.Swin       = np.zeros(tsteps)    # incoming short wave radiation [W m-2]
        self.Swout      = np.zeros(tsteps)    # outgoing short wave radiation [W m-2]
        self.Lwin       = np.zeros(tsteps)    # incoming long wave radiation [W m-2]
        self.Lwout      = np.zeros(tsteps)    # outgoing long wave radiation [W m-2]
        self.Q          = np.zeros(tsteps)    # net radiation [W m-2]

        # land surface variables
        self.ra         = np.zeros(tsteps)    # aerodynamic resistance [s m-1]
        self.rs         = np.zeros(tsteps)    # surface resistance [s m-1]
        self.H          = np.zeros(tsteps)    # sensible heat flux [W m-2]
        self.LE         = np.zeros(tsteps)    # evapotranspiration [W m-2]
        self.LEliq      = np.zeros(tsteps)    # open water evaporation [W m-2]
        self.LEveg      = np.zeros(tsteps)    # transpiration [W m-2]
        self.LEsoil     = np.zeros(tsteps)    # soil evaporation [W m-2]
        self.LEpot      = np.zeros(tsteps)    # potential evaporation [W m-2]
        self.LEref      = np.zeros(tsteps)    # reference evaporation at rs = rsmin / LAI [W m-2]
        self.G          = np.zeros(tsteps)    # ground heat flux [W m-2]

        # Raquel's additions
        self.Tsoil        = np.zeros(tsteps)  # temperature top soil layer [K]
        self.ETcommonpart = np.zeros(tsteps)  # common part of soil, vegetation and liquid latent heaat [W/m2]
        self.LE_radiation = np.zeros(tsteps)  # LE part related with the available energy [W/m2]
        self.LE_VPD       = np.zeros(tsteps)  # LE part related with the VPD [W/m2]
        self.LE_ideal     = np.zeros(tsteps)  # LE part taking into account qsatsurf (no EBC explicitly assumption)
        self.R            = np.zeros(tsteps)  # total resistance of an ecosystem [s m-1]
        self.gsoil_eff    = np.zeros(tsteps)  # effective soil conductance (it takes into account portion of soil) [m/s]
        self.gveg_eff     = np.zeros(tsteps)  # effective vegetation conductance [m/s]
        self.gliq_eff     = np.zeros(tsteps)  # effective liquid conductance [m/s]
        self.rssoil       = np.zeros(tsteps)  # soil resistance [s m-1]
        self.Ts           = np.zeros(tsteps)  # skin temperature [K]
        self.dqsatdT      = np.zeros(tsteps)  # material derivative of saturation humidity with respect to temperature [kg water/kg air/K]
        self.D0           = np.zeros(tsteps)  # VPD when stomata close [Pa]
        self.Ds           = np.zeros(tsteps)  # Leaf to air vapor pressure defictit [Pa]
        self.PAR          = np.zeros(tsteps)  # Photosynthetic active radiation [W/m2]
        # soil
        self.wg           = np.zeros(tsteps)  # volumetric water content top soil layer [m3 m-3]
        self.w2           = np.zeros(tsteps)  # volumetric water content deeper soil layer [m3 m-3]
        # ags
        self.co2abs       = np.zeros(tsteps)  # atmospheric CO2 concentration (mg CO2/m3)
        self.ci           = np.zeros(tsteps)  # CO2 concentration in the interior of the plant (mg CO2/m3)
        self.Ammax        = np.zeros(tsteps)  # CO2 maximal primary production (mg/m2leaf/s)
        self.Am           = np.zeros(tsteps)  # CO2 primary productivity (mg/m2leaf/s)
        self.Rdark        = np.zeros(tsteps)  # Dark respiration (mg/m2leaf/s)
        self.Ag           = np.zeros(tsteps)  # CO2 gross primary productivity at leaf level (mg/m2leaf/s)
        self.Agstar       = np.zeros(tsteps)  # Soil water unstressed CO2 gross primary productivity at leaf level (mg/m2leaf/s)
        self.Ag1          = np.zeros(tsteps)  # CO2 gross primary productivity at canopy (mg/m2/s)
        self.An           = np.zeros(tsteps)  # net flux of CO2 into the plant (mg/m2/s) 
        self.fstr         = np.zeros(tsteps)  # Plant water stress factor (0..1) [-]
        self.PCbeta       = np.zeros(tsteps)  # Plant water stress function to calculate the factor (intermediate step)
        self.LEleaf       = np.zeros(tsteps)  # Latent heat flux at leaf level [W/m2] NOT SURE REVISE
        self.glw          = np.zeros(tsteps)  # Stomatal conductancd of water at leaf level [m/s]
        self.glc          = np.zeros(tsteps)  # Stomatal conductancd of carbon dioxide at leaf level [m/s]
        self.a1           = np.zeros(tsteps)  # Intermediate parameter to calculate stomata conductance (-)
        self.alphac       = np.zeros(tsteps)  # Light use efficiency (mg J-1)
        self.CO2comp      = np.zeros(tsteps)  # CO2 compensation point (mg/m3) In CLASS book it says units are (ppmv(?))
        self.Dstar        = np.zeros(tsteps)  # Intermediate variable to calculate stomata conductance (Pa)
        self.fmin         = np.zeros(tsteps)  # Minimum value of Cfrac (-)
        self.fmin0        = np.zeros(tsteps)  # Intermediate conductance to calculate fmin (m/s)
        self.gm           = np.zeros(tsteps)  # Mesophyll conductance (m/s)
        self.cfrac        = np.zeros(tsteps)  # Fraction of the concentrations (ci-gamma)/(co2abs-gamma) (-)
        self.Aln          = np.zeros(tsteps)  # Net flow of CO2 into the plant per unit leaf area (mg CO2/m^2leaf/s)
        self.glc2_3       = np.zeros(tsteps)  # Stomatal conductancd of carbon dioxide at leaf level [m/s] at 2/3 canopy heights
        self.glc1_3       = np.zeros(tsteps)  # Stomatal conductancd of carbon dioxide at leaf level [m/s] at 1/3 canopy heights
        self.glc0         = np.zeros(tsteps)  # Stomatal conductancd of carbon dioxide at leaf level [m/s] at the ground
        ### Q: I think that according to theory An1 should be equal to An but I will check the difference. In the code they seem to be different
        self.Resp         = np.zeros(tsteps)  # CO2 soil surface flux (mg/m2/s)
        self.wstar        = np.zeros(tsteps)
        # Farquhar von Caemmerer and Berry (An), Medlyn (glc)
        
        ## end Raquel's additions ##
        
        # Mixed-layer top variables
        self.zlcl       = np.zeros(tsteps)    # lifting condensation level [m]
        self.RH_h       = np.zeros(tsteps)    # mixed-layer top relative humidity [-]

        # cumulus variables
        self.ac         = np.zeros(tsteps)    # cloud core fraction [-]
        self.M          = np.zeros(tsteps)    # cloud core mass flux [m s-1]
        self.dz         = np.zeros(tsteps)    # transition layer thickness [m]

# class for storing mixed-layer model input data
class model_input:
    def __init__(self):
        # general model variables
        self.runtime    = None  # duration of model run [s]
        self.dt         = None  # time step [s]

        # mixed-layer variables
        self.sw_ml      = None  # mixed-layer model switch
        self.sw_shearwe = None  # Shear growth ABL switch
        self.sw_fixft   = None  # Fix the free-troposphere switch
        self.h          = None  # initial ABL height [m]
        self.Ps         = None  # surface pressure [Pa]
        self.divU       = None  # horizontal large-scale divergence of wind [s-1]
        self.fc         = None  # Coriolis parameter [s-1]
        
        self.theta      = None  # initial mixed-layer potential temperature [K]
        self.dtheta     = None  # initial temperature jump at h [K]
        self.gammatheta = None  # free atmosphere potential temperature lapse rate [K m-1]
        self.advtheta   = None  # advection of heat [K s-1]
        self.beta       = None  # entrainment ratio for virtual heat [-]
        self.wtheta     = None  # surface kinematic heat flux [K m s-1]
        
        self.q          = None  # initial mixed-layer specific humidity [kg kg-1]
        self.dq         = None  # initial specific humidity jump at h [kg kg-1]
        self.gammaq     = None  # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.advq       = None  # advection of moisture [kg kg-1 s-1]
        self.wq         = None  # surface kinematic moisture flux [kg kg-1 m s-1]

        self.CO2        = None  # initial mixed-layer potential temperature [K]
        self.dCO2       = None  # initial temperature jump at h [K]
        self.gammaCO2   = None  # free atmosphere potential temperature lapse rate [K m-1]
        self.advCO2     = None  # advection of heat [K s-1]
        self.wCO2       = None  # surface kinematic heat flux [K m s-1]
        
        self.sw_wind    = None  # prognostic wind switch
        self.u          = None  # initial mixed-layer u-wind speed [m s-1]
        self.du         = None  # initial u-wind jump at h [m s-1]
        self.gammau     = None  # free atmosphere u-wind speed lapse rate [s-1]
        self.advu       = None  # advection of u-wind [m s-2]

        self.v          = None  # initial mixed-layer u-wind speed [m s-1]
        self.dv         = None  # initial u-wind jump at h [m s-1]
        self.gammav     = None  # free atmosphere v-wind speed lapse rate [s-1]
        self.advv       = None  # advection of v-wind [m s-2]

        # surface layer variables
        self.sw_sl      = None  # surface layer switch
        self.ustar      = None  # surface friction velocity [m s-1]
        self.z0m        = None  # roughness length for momentum [m]
        self.z0h        = None  # roughness length for scalars [m]
        self.Cm         = None  # drag coefficient for momentum [-]
        self.Cs         = None  # drag coefficient for scalars [-]
        self.L          = None  # Obukhov length [-]
        self.Rib        = None  # bulk Richardson number [-]

        # radiation parameters
        self.sw_rad     = None  # radiation switch
        self.lat        = None  # latitude [deg]
        self.lon        = None  # longitude [deg]
        self.doy        = None  # day of the year [-]
        self.tstart     = None  # time of the day [h UTC]
        self.cc         = None  # cloud cover fraction [-]
        self.Q          = None  # net radiation [W m-2] 
        self.dFz        = None  # cloud top radiative divergence [W m-2] 

        # land surface parameters
        self.sw_ls      = None  # land surface switch
        self.sw_gsdelay = None  # stomatal conductance delay (True or False)
        self.ls_type    = None  # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs or 'FvCB_M')
        self.wg         = None  # volumetric water content top soil layer [m3 m-3]
        self.w2         = None  # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil      = None  # temperature top soil layer [K]
        self.T2         = None  # temperature deeper soil layer [K]
        
        self.a          = None  # Clapp and Hornberger retention curve parameter a
        self.b          = None  # Clapp and Hornberger retention curve parameter b
        self.p          = None  # Clapp and Hornberger retention curve parameter p 
        self.CGsat      = None  # saturated soil conductivity for heat
        
        self.wsat       = None  # saturated volumetric water content ECMWF config [-]
        self.wfc        = None  # volumetric water content field capacity [-]
        self.wwilt      = None  # volumetric water content wilting point [-]
        
        self.C1sat      = None 
        self.C2ref      = None

        self.c_beta     = None  # Curvatur plant water-stress factor (0..1) [-]
        
        self.LAI        = None  # leaf area index [-]
        self.gD         = None  # correction factor transpiration for VPD [-]
        self.rsmin      = None  # minimum resistance transpiration [s m-1]
        self.rssoilmin  = None  # minimum resistance soil evaporation [s m-1]
        self.alpha      = None  # surface albedo [-]
        
        self.Ts         = None  # initial surface temperature [K]
        
        self.cveg       = None  # vegetation fraction [-]
        self.Wmax       = None  # thickness of water layer on wet vegetation [m]
        self.Wl         = None  # equivalent water layer depth for wet vegetation [m]
        
        self.Lambda     = None  # thermal diffusivity skin layer [-]

        # A-Gs parameters
        self.c3c4       = None  # Plant type ('c3' or 'c4')
        self.gsdelay    = None  # Stomatal delay scheme
        self.CO2comp298 =  [68.5,    4.3]   # CO2 compensation concentration (mgCO2/kg air)
        self.Q10CO2     =  [1.5,     1.5    ]   # function parameter to calculate CO2 compensation concentration [-]
        self.gm298      =  [7.0,     17.5   ]   # mesophyill conductance at 298 K [mm s-1]
        self.Ammax298   =  [2.2,     1.7    ]   # CO2 maximal primary productivity [mg m-2 s-1]
        self.Q10gm      =  [2.0,     2.0    ]   # function parameter to calculate mesophyll conductance [-]
        self.T1gm       =  [278.,    286.   ]   # reference temperature to calculate mesophyll conductance gm [K]
        self.T2gm       =  [301.,    309.   ]   # high reference temperature to calculate mesophyll conductance gm [K]
        self.Q10Am      =  [2.0,     2.0    ]   # function parameter to calculate maximal primary profuctivity Ammax
        self.T1Am       =  [281.,    286.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.T2Am       =  [311.,    311.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.f0         =  [0.89,    0.85   ]   # maximum value Cfrac [-]
        self.ad         =  [0.07,    0.15   ]   # regression coefficient to calculate Cfrac [kPa-1]
        self.alpha0     =  [0.017,   0.014  ]   # initial low light conditions [mg J-1]
        self.Kx         =  [0.7,     0.7    ]   # extinction coefficient PAR [-]
        self.gmin       =  [0.25e-3, 0.25e-3]   # cuticular (minimum) conductance to water [m s-1]
        self.taug       = None                  # time cosntant of stomatal response (s)

        # Farquhar-vonCaemmerer-Berry and Medlyn parameters
        self.Vcmax298   = 197*1e-6#LIAISE values 98e-6                 # maximum velocities of the carboxylase at 298 K (mol CO2/m2/s)
        self.Jmax298    = 398.63*1e-6
        self.Kc298      = 460e-6                # Michaelis-Menten constant for CO2 at 298 K (mol CO2/ mol air)
        self.Ko298      = 330e-3                # Michaelis-Menten constant for O2 at 298 K (mol O2/mol air)
        self.f_         = 0.15                  # Fraction of light lost as absorption by other than the chloroplast lamellae (-)
        self.CO2comp298_= 45e-6                 # Compensation point at 298 K (mol CO2/mol air)
        self.Oi         = 210*1e-3              # Leaf interior oxygen concentration (mol O2/mol air)
        self.g0_        = 0.033                 # minimum stomtatal conductance (mol air/m2/s)
        self.g1_        = 4.5#1.66                  # parameter to calculate stomata conductance (kPa-1/2)
        self.Kx_        = 0.7                   # extinction coefficient PAR [-]
        
        
        
        # Cumulus parameters
        self.sw_cu      = None  # Cumulus parameterization switch
        self.dz_h       = None  # Transition layer thickness [m]
        
        # Time or height dependent properties (currently both time and height dependency is not supported
        self.timedep    = None  # Time dependent variables
        self.heightdep  = None  # Height dependent variables
        
        
        