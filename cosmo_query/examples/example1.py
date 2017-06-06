#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:28:37 2017

@author: wolfensb
"""


import numpy as np
import matplotlib.pyplot as plt

from cosmo_query import config, SSH, Query
from cosmo_query import save_netcdf, load_netcdf
from cosmo_query import extract, coords_profile
##
# We initiate a connection to ela.cscs.ch, with username and password
# specified in the config.py file (password is not needed if ssh key is 
# defined)
connection = SSH(config.ELA_ADRESS,'wolfensb')
#
# We also need to open a channel to kesch.cscs.ch since it 
connection.open_channel(config.KESCH_ADRESS,'wolfensb')

# Now that the connection is setup we can create a Query instance
query = Query(connection)

# And use this query to retrieve some data

variables = ['T','P','QV'] # we want temperature and pressure
date = '2016-05-31 12:30' # for the 31th May 2016 at 12h30
model_res = 'fine' # at high resolution
mode = 'analysis' # In analysis mode
coord_bounds = ([6.6,45.8],[8.4,46.6]) # Over an area covering roughly the Valais

data = query.retrieve_data(variables, date, model_res = 'fine', 
                          mode = 'analysis', coord_bounds = coord_bounds)
#
## We can save this data to a netcdf
#save_netcdf(data,'myfile.nc')

# And load it again from this file
data = load_netcdf('myfile.nc')

# We can then use this data to extract a profile, here are some examples

# 1) Temperature at 2000 and 4000 m
prof1 = extract(data,['T'],'level',[2000,4000])

# 2) Latitudinal profile at lat = 46.2 degrees
prof2 = extract(data,['T'],'lat',46.2)

# 3) Longitudinal profile at lon = 7 degrees
prof3 = extract(data,['T'],'lon',7)

# 4) Transect going from Martigny to Visp with a step of 2000 m
coords = coords_profile([7.07,46.1],[7.88,46.29],step=2000)
# Note that you could create your own on straight transect as a N x 2 array of
# lon|lat coordinates in WGS84
prof3 = extract(data,['T'],'lonlat',coords)

# 5) PPI profile at 10 degree elevation from Martigny
options_PPI = {}
options_PPI['beamwidth'] = 1.5 # 1.5 deg 3dB beamwidth, as MXPol
options_PPI['elevation'] = 10
options_PPI['rrange'] = np.arange(200,20000,75) # from 0.2 to 20 km with a res of 75 m
options_PPI['npts_quad'] = [3,3] # 3 quadrature points in azimuthal, 3 in elevational directions
options_PPI['rpos'] = [7.0923,46.1134,500] # Radar position in lon/lat/alt
options_PPI['refraction_method'] = 1 # 1 = standard 4/3, 2 = ODE refraction 
# model : more accurate but needs P (pressure), QV (water vapour) and 
# T (temperature) to be in the data structure

# Note that the COSMO DEM is quite rough, so the topography of COSMO might 
# be higher than the real topography...for example real altitude of this point
# is 458.2 but closest COSMO point is at 626.28 m altitude...

ppi_T = extract(data,['T'],'PPI',options_PPI) 

# Missing points correspond to areas where either all  the signal is lost 
# (beam hit a mountain) or is located above COSMO domain

# 5) RHI profile at 47 degree azimuth
options_RHI = {}
options_RHI['beamwidth'] = 1.5 # 1.5 deg 3dB beamwidth, as MXPol
options_RHI['azimuth'] = 47
options_RHI['rrange'] = np.arange(200,10000,75) # from 0.2 to 10 km with a res of 75 m
options_RHI['npts_quad'] = [3,3] # 3 quadrature points in azimuthal, 3 in elevational directions
options_RHI['rpos'] = [7.0923,46.1134,500] # Radar position in lon/lat/alt
options_RHI['refraction_method'] = 1 # 1 = standard 4/3, 2 = ODE refraction 

rhi_T = extract(data,['T'],'RHI',options_RHI) 