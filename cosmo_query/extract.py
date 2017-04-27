from collections import OrderedDict
import numpy as np
from metaarray import MetaArray

import pyproj as pyproj
import numpy as np
import scipy.spatial as spatial

from beam_tracing import global_to_local_coords
from beam_tracing import _Beam, refraction_sh, quad_pts_weights, integrate_quad
from c.radar_interp_c import get_all_radar_pts

def get_refractivity(data):
    """
    FUNCTION:
        get_refractivity(data)
    
    PURPOSE:
         Compute the atmospheric refractivity profile from COSMO data
    
    INPUTS:
        data : a dictionary with three fields 'P' : the atm. pressure, 'T':
            the temperature, 'QV' : the water vapour concentration
    OUTPUTS:
        N : the estimated atmospheric refractivity
    
    """    
    Pw = (data['P'] * data['QV']) / (data['QV'] * (1 - 0.6357) + 0.6357)
    N = (77.6 / data['T']) * (0.01 * data['P'] + 4810 * (0.01 * Pw) / data['T'])
    
    mapping = data['P'].metadata['mapping']
    x = data['x_'+str(mapping)]
    y = data['y_'+str(mapping)]
    
    # We also need to add some info about the coordinate system of N
    
    # Get lower left corner of COSMO domain in local coordinates
    llc_COSMO = [x[0],y[0]]
    llc_COSMO = np.array(llc_COSMO).astype('float32')

    # Get resolution             
    res_COSMO = [x[1] - x[0], y[1] - y[0]]

    # Get latitude and longitude of southern pole
    grid_mapping = data['grid_mapping_' + str(mapping)]
    lat_SP = - grid_mapping.metadata['grid_north_pole_latitude']
    lon_SP = grid_mapping.metadata['grid_north_pole_longitude'] - 180
  
    N.attributes = {}
    N.attributes['proj_info'] = {'Latitude_of_southern_pole':lat_SP,
                                'Longitude_of_southern_pole':lon_SP,
                                'Lo1':llc_COSMO[0],
                                'La1':llc_COSMO[1]}
    
    N.attributes['resolution'] = res_COSMO
    return N

def coords_profile(start, stop, step = None, npts = None):
    """
    FUNCTION:
        coords_profile(start, stop, step=-1, npts=-1)
    
    PURPOSE:
        Computes a linear profile of coordinates going from point a to point b
        with a either a fixed step distance or a fixed number of points
    
    INPUTS:
        start : a tuple specifying the coordinates of point a in (lon,lat) 
            WGS84 coordinates
        stop : a tuple specifying the coordinates of point b in (lon,lat) 
            WGS84 coordinates
        step : (optional) the distance between two consecutive points (in m.)
        npts : (optional) the total number of points along the profile, inclu-
            ding start and end points
        
        NOTE that you must provide either step or npts!
        
    OUTPUTS:
        profile : a N x 2 array of lon/lat coordinates, N being the total 
                  number of points
    
    """    
    
    start=np.asarray(start)
    stop=np.asarray(stop)
    use_step=False
    use_npts=False
    
    # Check inputs
    if not step and npts < 3:
        msg = """
        Neither a valid distance step nor a valid number of points of the 
        transect have been specified, please provide one or the other!
        
        Number of points must be larger than 3 and step distance must 
        be larger than 0
        """
        raise ValueError(msg)
    elif step and npts >= 3:
        msg = """
        Both a distance step and a number of points in the transect have been 
        specified, only the distance step will be used!
        """
        raise ValueError(msg)
    elif step and not npts:
        use_step=True
    else:
        use_npts=True
        
    g = pyproj.Geod(ellps='WGS84') # Use Clarke 1966 ellipsoid.
    az12,az21,dist = g.inv(start[0],start[1],stop[0],stop[1]) # Backward transform
    
    if use_step:
        npts = np.floor(dist/step)
        dist = npts*step
        endlon, endlat, backaz = g.fwd(start[0],start[1],az12, dist)
        profile = g.npts(start[0],start[1], endlon, endlat ,npts-2)
    if use_npts:
        profile = g.npts(start[0],start[1],stop[0],stop[1],npts-2)
    
    # Add start points
    profile.insert(0,(start[0],start[1]))
    
    if use_npts:
        profile.insert(len(profile),(stop[0],stop[1]))
        
    profile = np.asarray(profile)
    return profile

def extract(data, variable_names, slice_type, idx, parallel = False):
    """
    FUNCTION:
        extract(data, variable_names, slice_type, idx, parallel = False)
    
    PURPOSE:
        Extracts a specific profile of COSMO data along various types of 
        transects using linear interpolation
    
    INPUTS:
        data : a data structure containing COSMO data as returned by a COSMO
            query
            
        variable_names : the list of variables to be extracted along the pro-
            file, they need to be included in data!
            
        slice_type : the type of slice to be extracted, need to be either
            'level', 'lon', 'lat', 'lonlat', 'PPI' or 'RHI'
            
        idx : information about how and where the slice should be done, see
            more specific info in corresponding part of the code
            
        parallel : TODO allows parallel computation (not implemented yet)
        
    OUTPUTS:
        output : a dictionary containing one array for every retrieved variable
            these arrays have additional fields 'coordinates' which contains
            the coordinates of the data and for some slice types 'attributes'
            which contains additional useful data
    
    """    
    
    if slice_type not in ['level','lon','lat','lonlat','PPI','RHI']:
        msg = """
        Invalid profile type, choose either level, lon, lat, lonlat, PPI or RHI
        """
        raise ValueError(msg)
        
    if not isinstance(variable_names,list):
        variable_names =  [variable_names]
    
    all_dimensions = np.array([data[var].ndim for var in variable_names])
    all_mappings = [data[var].metadata['mapping'] for var in variable_names]
    all_ztypes = [data[var].metadata['ztype'] for var in variable_names]
    
    
    if np.any(all_dimensions != 3) and slice_type in ['level','PPI','RHI']:
        raise ValueError('Variable must be 3D to cut at vertical coordinates!')

    storage = {} # This is used to store some data that is consistent for the 
    # different variables
    output = {}
    
    if slice_type == 'level':
        """
        slice_type = 'level'
        
        PURPOSE : 
            Cut plane(s) at fixed altitude levels (with linear interpolation)

        INPUTS :
            idx : a list or array of altitude levels, ex idx = [1000,2000,5000]
        
        """
        cut_val = idx
            
        if type(cut_val) != list:
            cut_val = np.array([cut_val])
            
        for i,varname in enumerate(variable_names):
            data_interp = []
            
            var = data[varname]
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]

            key = str(mapping) + '_' + str(ztype)
            
            if key in storage:
                zlevels = storage[key][0]
                indexes_larger = storage[key][1]
                indexes_smaller = storage[key][2]
            else:
                indexes_larger = []
                indexes_smaller = []
                
                zlevels = data['z_'+str(mapping)]      
                if ztype == 'half':
                    zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])
                
                for cv in cut_val:
                    diff = (zlevels-cv)
                    diff[diff<0] = 99999 # Do this in order to find only the indexes of heights > cut_val
                    
                    idx_larg = np.argmin(diff,axis=0)
                    idx_smal = idx_larg + 1 # remember that vert. coords are in decreasing order...
                    idx_smal[idx_smal>siz[0]-1] = siz[0] - 1
                    
                    indexes_larger.append(idx_larg)
                    indexes_smaller.append(idx_smal) 
                    
                storage[key] = []
                storage[key].append(zlevels)
                storage[key].append(indexes_larger)
                storage[key].append(indexes_smaller)
                
            # One of numpy's limitation is that we have to index with these index arrays
            k,j = np.meshgrid(np.arange(siz[2]),np.arange(siz[1])) 
            
            for i,cv in enumerate(cut_val):
                dist = zlevels[indexes_larger[i],j,k]-zlevels[indexes_smaller[i],j,k]
            
                # Perform linear interpolation
                dinterp = var[:][indexes_smaller[i],j,k]+(var[:][indexes_larger[i],j,k]-\
                    var[:][indexes_smaller[i],j,k])/dist*(cv-zlevels[indexes_smaller[i],j,k])
           
                # Now mask
                dinterp[indexes_smaller[i]==siz[0]-1]=float('nan')
                data_interp.append(dinterp)
            
            data_interp = np.array(data_interp)
            marray = MetaArray(data_interp)
            coords = OrderedDict()
            coords['heights'] = cut_val
            coords['lat'] = data['lat_'+str(mapping)] 
            coords['lon'] = data['lon_'+str(mapping)] 

            marray.coordinates = coords
            
            output[varname] = marray
        
    if slice_type == 'lat':
        """
        slice_type = 'lat'
        
        PURPOSE : 
            Cut transect at fixed latitude (in WGS84 coordinates)

        INPUTS :
            idx : a float with the latitude, ex idx = 48.5
        
        """
        
        # First we check if the given profile is valid
        for i,varname in enumerate(variable_names):

            var = data[varname]
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]
  
            # Get latitudes and longitudes
            lat_2D = data['lat_'+str(mapping)] 
            lon_2D = data['lon_'+str(mapping)] 

            if np.max(lat_2D) < idx or np.min(lat_2D) > idx:
                raise ValueError('Specified latitude is outside of available domain')
                
            key = str(mapping) + '_' + str(ztype)

            if key in storage:
                zlevels = storage[key][0]
                indices_lon = storage[key][1]
                slice_lon = storage[key][2]
            else:
                # Get dimensions and zlevels
                siz = var.shape
                if var.ndim == 3:
                    n_lon = siz[2]
                    n_vert = siz[0]
                    zlevels = data['z_'+str(mapping)]      
                    if ztype == 'half':
                        zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])
                else: # variable is 2D
                    n_lon = siz[1]
                    n_vert = 1

                
                # Get indices of longitude for every column
                # Initialize
                indices_lon = np.zeros((n_lon,2),dtype=int) 
                slice_lon = np.zeros((n_lon,))
                for i in range(0,n_lon): # Loop on all longitudes
                    # Get the idx for the specified latitude
                    idx_lat = int(np.argmin(np.abs(lat_2D[:,i]-idx)))
                    indices_lon[i,:] = [idx_lat,i]
                    slice_lon[i] = lon_2D[idx_lat,i]
            
                storage[key] = []
                storage[key].append(zlevels)
                storage[key].append(indices_lon)
                storage[key].append(slice_lon)
                
            # Now get data at indices for every vertical layer  

            # Initialize interpolated values
            data_interp = np.zeros((n_vert,n_lon))
            if var.ndim == 3:
                for j in range(0, n_vert):
                    data_interp[j,:] = np.array([var[j,pt[0],pt[1]] \
                                                for pt in indices_lon])
            elif var.ndim == 2:
                data_interp = np.array([var[pt[0],pt[1]] \
                                          for pt in indices_lon])
            
            # If variable has z-levels, get them along profile
            if var.ndim == 3:
                zlevels_slice = np.zeros((n_vert, n_lon))
                for j in range(0, n_vert):
                    zlevels_slice[j,:] = [zlevels[j, pt[0], pt[1]] \
                            for pt in indices_lon]
                
            marray = MetaArray(data_interp)
            coords = OrderedDict()
            coords['heights'] = zlevels_slice
            coords['lon/lat'] = np.vstack((slice_lon,
                                      np.ones(len(slice_lon)) * idx)).T

            marray.coordinates = coords
            
            output[varname] = marray

    if slice_type == 'lon':
        """
        slice_type = 'lon'
        
        PURPOSE : 
            Cut transect at fixed longitude (in WGS84 coordinates)

        INPUTS :
            idx : a float with the longitude, ex idx = 11
        
        """
        # First we check if the given profile is valid
        for i,varname in enumerate(variable_names):

            var = data[varname]
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]
  
            # Get latitudes and longitudes
            lat_2D = data['lat_'+str(mapping)] 
            lon_2D = data['lon_'+str(mapping)] 
            
            if np.max(lon_2D) < idx or np.min(lon_2D) > idx:
                raise ValueError('Specified longitude is outside of available domain')
                
            key = str(mapping) + '_' + str(ztype)

            if key in storage:
                zlevels = storage[key][0]
                indices_lat = storage[key][1]
                slice_lat = storage[key][2]
            else:
                # Get dimensions and zlevels
                siz = var.shape
                if var.ndim == 3:
                    n_lat = siz[1]
                    n_vert = siz[0]
                    zlevels = data['z_'+str(mapping)]      
                    if ztype == 'half':
                        zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])
                else: # variable is 2D
                    n_lat = siz[0]
                    n_vert = 1

                
                # Get indices of longitude for every column
                # Initialize
                indices_lat = np.zeros((n_lat,2),dtype=int) 
                slice_lat = np.zeros((n_lat,))
                for i in range(0,n_lat): # Loop on all latitudes
                    # Get the idx for the specified latitude
                    idx_lon = int(np.argmin(np.abs(lon_2D[:,i]-idx)))
                    indices_lat[i,:] = [i,idx_lon]
                    slice_lat[i] = lat_2D[i,idx_lon]
            
                storage[key] = []
                storage[key].append(zlevels)
                storage[key].append(indices_lat)
                storage[key].append(slice_lat)
                
            # Now get data at indices for every vertical layer  

            # Initialize interpolated values
            data_interp = np.zeros((n_vert,n_lat))
            if var.ndim == 3:
                for j in range(0, n_vert):
                    data_interp[j,:] = np.array([var[j,pt[0],pt[1]] \
                                                for pt in indices_lat])
            elif var.ndim == 2:
                data_interp = np.array([var[pt[0],pt[1]] \
                                          for pt in indices_lat])
            
            # If variable has z-levels, get them along profile
            if var.ndim == 3:
                zlevels_slice = np.zeros((n_vert, n_lat))
                for j in range(0, n_vert):
                    zlevels_slice[j,:] = [zlevels[j, pt[0], pt[1]] \
                            for pt in indices_lat]
                
            marray = MetaArray(data_interp)
            coords = OrderedDict()
            
            coords['heights'] = zlevels_slice
            coords['lon/lat'] = np.vstack((np.ones(len(slice_lat)) * idx,
                                          slice_lat)).T

            marray.coordinates = coords
            
            output[varname] = marray
        
    if slice_type == 'lonlat':
        """
        slice_type = 'lonlat'
        
        PURPOSE : 
            Cut transect at fixed longitude/latitude coordinates in WGS84

        INPUTS :
            idx : a Nx2 float array containing longitudes in the first 
                column and latitudes in the second column
        
        """
        for i,varname in enumerate(variable_names):

            var = data[varname]
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]
  
            # Get latitudes and longitudes of data
            lat_2D = data['lat_'+str(mapping)] 
            lon_2D = data['lon_'+str(mapping)] 
            
            # Get latitudes and longitudes of desired profile
            lat_prof = idx[:,1]
            lon_prof = idx[:,0]
            n_pts = len(idx)
            
            # Check if all profile points are within domain
            if np.any(lon_prof > np.max(lon_2D)) or np.any(lon_prof < np.min(lon_2D)) or \
                np.any(lat_prof > np.max(lat_2D)) or np.any(lat_prof < np.min(lat_2D)):
                raise ValueError('Specified profile is outside available domain')
                
            key = str(mapping) + '_' + str(ztype)

            if key in storage:
                zlevels = storage[key][0]
                indices_lat = storage[key][1]
                slice_lat = storage[key][2]
                slice_lon = storage[key][3]
                dist_along_prof = storage[key][4]
            else:
                
                lat_2D_stack = lat_2D.ravel()
                lon_2D_stack = lon_2D.ravel()
                # Get array of all latitudes and longitudes of COSMO grid points
                combined_latlon = np.dstack([lat_2D_stack,lon_2D_stack])[0]
                
                # Create kd-tree of all COSMO points
                tree = spatial.cKDTree(combined_latlon)
                # Now get indexes of profile points
                dist, indexes = tree.query(idx)
                
                # Get latitude and longitudes corresponding to indexed
                slice_lat = lat_2D_stack[indexes]
                slice_lon = lon_2D_stack[indexes]
            
                 # Now we get the distance along the profile as the new coordinates
                g = pyproj.Geod(ellps='WGS84') # Use WGS84 ellipsoid.
                
                all_dist_pairs = [0]
                for i in range(len(slice_lon) - 1):
                    _,_,d = g.inv(lon_prof[i],lat_prof[i],
                                       lon_prof[i+1],lat_prof[i+1]) # Backward transform
                    all_dist_pairs.append(d)
                    
                dist_along_prof = np.cumsum(np.array(all_dist_pairs))
                # If variable is 3D get zlevels as well
                if var.ndim == 3:
                    n_vert = siz[0]
                    zlevels = data['z_'+str(mapping)]      
                    if ztype == 'half':
                        zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])
                else: # variable is 2D
                    n_vert = 1
                
                # Put everything important into storage dict
                storage[key] = []
                storage[key].append(zlevels)
                storage[key].append(indexes)
                storage[key].append(slice_lat)
                storage[key].append(slice_lon)
                storage[key].append(dist_along_prof)
                
            # Now get data at indices for every vertical layer  

            # Initialize interpolated values
            data_interp = np.zeros((n_vert,n_pts))
            if var.ndim == 3:
                for j in range(0, n_vert):
                    data_stack = var[j,:,:].ravel()
                    data_interp[j,:] = data_stack[indexes]
            elif var.ndim == 2:
                data_stack=var.ravel()
                data_interp = data_stack[indexes]
            
            # If variable has z-levels, get them along profile
            if var.ndim == 3:
                zlevels_slice = np.zeros((n_vert, n_pts))
                for j in range(0, n_vert):
                    z_stack = zlevels[j].ravel()
                    zlevels_slice[j,:] = z_stack[indexes]
                
            marray = MetaArray(data_interp)
            
            # Create coordinates
            coords = OrderedDict()
            coords['heights'] = zlevels_slice
            coords['lon/lat'] = np.vstack((lon_prof,lat_prof)).T

            # Create attributes
            attrs = OrderedDict()
            attrs['distance'] = dist_along_prof
            
            # Assign coordinates and attributes
            marray.coordinates = coords
            marray.attributes = attrs
            
            output[varname] = marray
                
    if slice_type == 'PPI':
        """
        slice_type = 'PPI'
        
        PURPOSE : 
            Cut plane along the coordinates of a PPI radar scan, using 
            atmospheric refraction and integration over the antenna diagram

        INPUTS :
            idx : a dictionary with the following fields :
                rrange : a list or array of floats giving the distance of every 
                    radar gate
                rpos : (lon,lat, altitude) 3D coordinates of the radar in 
                    WGS84 coordinates
                elevation : float specifying the elevation angle of the 
                    PPI scan   
                beamwidth : the 3dB beamwidth of the antenna, will not
                    be used if you set npts_quad = [1,1] (no antenna integ.)
                azimuths : (optional) a list or array of floats corresponding
                    to the azimuth angles of the PPI scan, if not set a default
                    list going from 0 to 360 with a step corresponding to the
                    beamwidth will be used
                npts_quad : (optional) a tuple giving the number of interpola-
                    tion points in azimuthal and elevational directions. The
                    default is (3,3)
                refraction_method : (optional) To compute the propagation of 
                    the radar beam, two schemes can be used: if 
                    refraction_scheme = 1, the standard 4/3 Earth radius scheme
                    will be used, if refraction_scheme = 2, a more accurate 
                    scheme based on a refractivity profile estimated from 
                    the model will be used; this requires QV (water vapour), 
                    T (temperature) and P (pressure) to be present in the file.
                    Default is 1
        """
        
        # Check scan specifications
        # Mandatory arguments
        try:
            # Read fields from the input dictionary
            rpos = [float(i) for i in idx['rpos']] # radar coords.
            elevation = float(idx['elevation']) # elevation angle  
            rrange = np.array(idx['rrange']) # distance of radar gates
            beamwidth_3dB = float(idx['beamwidth']) # 3dB antenna beamwidth
        except:
            msg = """
            In order to interpolate COSMO data along a radar RHI, you have 
            to give as second input a dictionary with mandatory fields:
                
                rpos : (lon,lat, altitude) 3D coordinates of the radar in 
                        WGS84 coordinates
                        
                elevation : double specifying the elevation angle of the 
                            PPI scan                
                            
                rrange : a list of floats giving  the distance of every 
                         radar gate
                         
                beamwidth :     the 3dB beamwidth of the antenna, will not
                                be used if you set npts_quad = [1,1] (no
                                antenna quadrature)
            """
            raise ValueError(msg)           
        # Optional arguments:
        try:
            azimuths = np.array(idx['azimuths'])
        except:
            msg = """
            Could not read 'azimuth' field in input dict. 
            Using default one: np.arange(0,360,beamwidth_3dB)
            """
            print(msg)
            azimuths = np.arange(0,360 + beamwidth_3dB, beamwidth_3dB)
    
        try:
            npts_quad = [int(i) for i in idx['npts_quad']]
        except:
            msg = """
            Could not read 'npts_quad' field in input dict. 
            Using default one: [5,3]
            """
            print(msg)
            npts_quad = [5,5]

        try:
            refraction_method = int(idx['refraction_method'])
        except:
            msg = """
            Could not read 'refraction_method' field in input dict.
            Using default one: 1 (4/3 Earth radius method)
            """
            print(msg)
            refraction_method = 1
    
        # If refraction_method == 2: try to get N (refractivity) 
        N = []
        if refraction_method == 2:
            try: 
                msg = """
                Trying to compute atm. refractivity (N) from COSMO file
                """
                print(msg)
                N = get_refractivity(data)
            except:
                pass
                refraction_method = 1
                msg = """
                Could not compute N from COSMO data
                Make sure that the variables P (pressure), T (temp),
                and QV (water vapour content) are in the provided data
                
                Using refraction_method = 1, instead'
                """
                print(msg)
                print('Could not compute N from COSMO data'+
                  ' Using refraction_method = 1, instead')
            
        # Get the quadrature points        
        pts,weights = quad_pts_weights(beamwidth_3dB, npts_quad)
        
        # Initialize WGS84 geoid
        g = pyproj.Geod(ellps='WGS84')
        
        # Precompute all radar beams coordinates (at they do not depend on grid
        # mapping)
        all_s = []
        all_h = []
        all_lons_radial = []
        all_lats_radial = []
        
        for radial,az in enumerate(azimuths): # Loop on all radials
            for j,pt_vert in enumerate(pts[1]): 
                s,h = refraction_sh(rrange,pt_vert+elevation,rpos,
                                    refraction_method, N)
                all_s.append(s)
                all_h.append(h)
                for i,pt_hor in enumerate(pts[0]):
                    # Get radar gate coordinates in WGS84
                    lons_radial = []
                    lats_radial = []
                    for s_gate in s:
                        lon,lat,ang = g.fwd(rpos[0],rpos[1], pt_hor + az, s_gate)  
                        lons_radial.append(lon)
                        lats_radial.append(lat)
                        
                    all_lons_radial.append(np.array(lons_radial))
                    all_lats_radial.append(np.array(lats_radial))
                    

        for i,varname in enumerate(variable_names):
            
            var = data[varname]
            
            # Initialize interpolated values and coords
            data_interp = np.zeros((len(rrange),len(azimuths)))
            lons_scan =  np.zeros((len(rrange),len(azimuths)))
            lats_scan = np.zeros((len(rrange),len(azimuths)))
            heights_scan = np.zeros((len(rrange),len(azimuths)))
            dist_ground = np.zeros((len(rrange),len(azimuths)))
            frac_power = np.zeros((len(rrange),len(azimuths)))
                
            if var.ndim != 3:
                msg = """ 
                In order to slice on a radar PPI the variable must be 3D,
                aborting...
                """
                raise ValueError(msg)   
        
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]
  
            # Get latitudes and longitudes of data
            lat_2D = data['lat_'+str(mapping)] 
            lon_2D = data['lon_'+str(mapping)] 
            
            
            # Get model heights and COSMO proj from first variable    
            zlevels = data['z_'+str(mapping)]      
            if ztype == 'half':
                zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])

            # Get COSMO local coordinates info
            x = data['x_'+str(mapping)]
            y = data['y_'+str(mapping)]
            # Get lower left corner of COSMO domain in local coordinates
            llc_COSMO = [x[0],y[0]]
            llc_COSMO = np.array(llc_COSMO).astype('float32')
            
            # Get upper left corner of COSMO domain in local coordinates
            urc_COSMO = [x[-1],y[-1]]
            urc_COSMO = np.array(urc_COSMO).astype('float32')
    
            # Get resolution             
            res_COSMO = [x[1] - x[0], y[1] - y[0]]

            # Get latitude and longitude of southern pole
            grid_mapping = data['grid_mapping_' + str(mapping)]
            lat_SP = - grid_mapping.metadata['grid_north_pole_latitude']
            lon_SP = grid_mapping.metadata['grid_north_pole_longitude'] - 180
            
            
            # Main loop
            global_idx = 0
            for radial,az in enumerate(azimuths): # Loop on all radials
                print('Processing azimuth ang. '+str(az))
                list_beams = []
                for j,pt_vert in enumerate(pts[1]): 
                    for i,pt_hor in enumerate(pts[0]):
                        
                        lats_radial = all_lats_radial[global_idx]
                        lons_radial = all_lons_radial[global_idx]
                        
                        # Transform radar gate coordinates into local COSMO coordinates
                        coords_rad_loc = global_to_local_coords(lons_radial,
                                             lats_radial,[lon_SP,lat_SP])  
    
                        # Check if all points are within COSMO domain
                        
                        if np.any(coords_rad_loc[:,0]<llc_COSMO[0]) or\
                            np.any(coords_rad_loc[:,1]<llc_COSMO[1]) or \
                            np.any(coords_rad_loc[:,0]>urc_COSMO[0]) or \
                            np.any(coords_rad_loc[:,1]>urc_COSMO[1]):
                            
                            
                            raise(IndexError('Radar domain is not entirely contained'+
                                                          ' in COSMO domain, aborting'))
                        
                        dic_beams = {}
                   
                        model_data=var[:]
                        
                        coords_rad_loc = np.fliplr(coords_rad_loc).astype('float32')
                        
                        rad_interp_values = get_all_radar_pts(len(rrange),\
                                          coords_rad_loc,h,model_data, \
                                          zlevels, llc_COSMO,res_COSMO)[1]
            
                 
                        mask_beam = np.zeros((len(rad_interp_values)))
                        mask_beam[rad_interp_values == -9999] = -1 # Means that the interpolated point is above COSMO domain
                        mask_beam[np.isnan(rad_interp_values)] = 1  # Means that the interpolated point is below COSMO terrain
                        rad_interp_values[mask_beam!=0] = np.nan # Assign NaN to all missing data
                        dic_beams[varname] = rad_interp_values
                        
                        list_beams.append(_Beam(dic_beams, mask_beam, lats_radial,
                            lons_radial, s,h,[pt_hor + az,pt_vert+elevation],
                            weights[i,j]))         
                    
                        # Increment global index
                        global_idx += 1
                            
                # Integrate all sub-beams
                scan, frac_pow = integrate_quad(list_beams)    
                
                # Add radial
                data_interp[:,radial] = scan.values[varname]
                lats_scan[:,radial] = scan.lats_profile
                lons_scan[:,radial] = scan.lons_profile
                heights_scan[:,radial] = scan.heights_profile
                dist_ground[:,radial] = scan.dist_profile
                frac_power[:,radial] = frac_pow
                
            marray = MetaArray(data_interp)
            
            # Create coordinates
            coords = OrderedDict()
            coords['rrange'] = rrange
            coords['azimuth'] = azimuths
            
            # Create attributes
            attrs = OrderedDict()
            attrs['heights'] = heights_scan
            attrs['lon'] = lons_scan
            attrs['lat'] = lats_scan
            attrs['elevation'] = elevation
            attrs['frac_power'] = frac_power
            
            # Assign coords/attrs
            marray.coordinates = coords
            marray.attributes = attrs
            
            # Assign to overall dict
            output[varname] = marray  
                
    if slice_type == 'RHI':
        """
        slice_type = 'RHI'
        
        PURPOSE : 
            Cut plane along the coordinates of a RHI scan, using 
            atmospheric refraction and integration over the antenna diagram

        INPUTS :
            idx : a dictionary with the following fields :
                rrange : a list or array of floats giving the distance of every 
                    radar gate
                rpos : (lon,lat, altitude) 3D coordinates of the radar in 
                    WGS84 coordinates
                azimuth : float specifying the azimuth angle of the 
                    RHI scan   
                beamwidth : the 3dB beamwidth of the antenna, will not
                    be used if you set npts_quad = [1,1] (no antenna integ.)
                elevations : (optional) a list or array of floats corresponding
                    to the elevation angles of the PPI scan, if not set a default
                    list going from 0 to 360 with a step corresponding to the
                    beamwidth will be used
                npts_quad : (optional) a tuple giving the number of interpola-
                    tion points in azimuthal and elevational directions. The
                    default is (3,3)
                refraction_method : (optional) To compute the propagation of 
                    the radar beam, two schemes can be used: if 
                    refraction_scheme = 1, the standard 4/3 Earth radius scheme
                    will be used, if refraction_scheme = 2, a more accurate 
                    scheme based on a refractivity profile estimated from 
                    the model will be used; this requires QV (water vapour), 
                    T (temperature) and P (pressure) to be present in the file.
                    Default is 1
        """
        # Check scan specifications
        # Mandatory arguments
        try:
            # Read fields from the input dictionary
            rpos = [float(i) for i in idx['rpos']] # radar coords.
            azimuth = float(idx['azimuth']) # azimuth angle  
            rrange = np.array(idx['rrange']) # distance of radar gates
            beamwidth_3dB = float(idx['beamwidth']) # 3dB antenna beamwidth
        except:
            msg = """
            In order to interpolate COSMO data along a radar RHI, you have 
            to give as second input a dictionary with mandatory fields:
                
                rpos : (lon,lat, altitude) 3D coordinates of the radar in 
                        WGS84 coordinates
                        
                azimuth : double specifying the azimuth angle of the 
                             RHI scan                
                            
                rrange : a list of floats giving  the distance of every 
                         radar gate
                         
                beamwidth :     the 3dB beamwidth of the antenna, will not
                                be used if you set npts_quad = [1,1] (no
                                antenna quadrature)
            """
            raise ValueError(msg)           
        # Optional arguments:
        try:
            elevations = np.array(idx['elevation'])
        except:
            msg = """
            Could not read 'elevation' field in input dict. 
            Using default one: np.arange(0,90,beamwidth_3dB)
            """
            print(msg)
            elevations = np.arange(0,90 + beamwidth_3dB, beamwidth_3dB)
    
        try:
            npts_quad = [int(i) for i in idx['npts_quad']]
        except:
            msg = """
            Could not read 'npts_quad' field in input dict. 
            Using default one: [5,3]
            """
            print(msg)
            npts_quad = [5,5]

        try:
            refraction_method = int(idx['refraction_method'])
        except:
            msg = """
            Could not read 'refraction_method' field in input dict.
            Using default one: 1 (4/3 Earth radius method)
            """
            print(msg)
            refraction_method = 1
    
        # If refraction_method == 2: try to get N (refractivity) 
        N = []
        if refraction_method == 2:
            try: 
                msg = """
                Trying to compute atm. refractivity (N) from COSMO file
                """
                print(msg)
                N = get_refractivity(data)
            except:
                pass
                refraction_method = 1
                msg = """
                Could not compute N from COSMO data
                Make sure that the variables P (pressure), T (temp),
                and QV (water vapour content) are in the provided data
                
                Using refraction_method = 1, instead'
                """
                print(msg)
                print('Could not compute N from COSMO data'+
                  ' Using refraction_method = 1, instead')
            
        # Get the quadrature points        
        pts,weights = quad_pts_weights(beamwidth_3dB, npts_quad)
        
        # Initialize WGS84 geoid
        g = pyproj.Geod(ellps='WGS84')
        
        # Precompute all radar beams coordinates (at they do not depend on grid
        # mapping)
        all_s = []
        all_h = []
        all_lons_radial = []
        all_lats_radial = []
        
        for radial,elev in enumerate(elevations): # Loop on all radials
            for j,pt_vert in enumerate(pts[1]): 
                s,h = refraction_sh(rrange,pt_vert+elev,rpos,
                                    refraction_method, N)
                all_s.append(s)
                all_h.append(h)
                for i,pt_hor in enumerate(pts[0]):
                    # Get radar gate coordinates in WGS84
                    lons_radial = []
                    lats_radial = []
                    for s_gate in s:
                        lon,lat,ang = g.fwd(rpos[0],rpos[1],
                                            pt_hor + azimuth, s_gate)  
                        lons_radial.append(lon)
                        lats_radial.append(lat)
                        
                    all_lons_radial.append(np.array(lons_radial))
                    all_lats_radial.append(np.array(lats_radial))
                    

        for i,varname in enumerate(variable_names):
            
            var = data[varname]
            
            # Initialize interpolated values and coords
            data_interp = np.zeros((len(rrange),len(elevations)))
            lons_scan =  np.zeros((len(rrange),len(elevations)))
            lats_scan = np.zeros((len(rrange),len(elevations)))
            heights_scan = np.zeros((len(rrange),len(elevations)))
            dist_ground = np.zeros((len(rrange),len(elevations)))
            frac_power = np.zeros((len(rrange),len(elevations)))
                
            if var.ndim != 3:
                msg = """ 
                In order to slice on a radar PPI the variable must be 3D,
                aborting...
                """
                raise ValueError(msg)   
        
            siz = var.shape
                           
            mapping = all_mappings[i]
            ztype = all_ztypes[i]
  
            # Get latitudes and longitudes of data
            lat_2D = data['lat_'+str(mapping)] 
            lon_2D = data['lon_'+str(mapping)] 
            
            
            # Get model heights and COSMO proj from first variable    
            zlevels = data['z_'+str(mapping)]      
            if ztype == 'half':
                zlevels = 0.5 * (zlevels[0:-1,:,:] + zlevels[1:,:,:])

            # Get COSMO local coordinates info
            x = data['x_'+str(mapping)]
            y = data['y_'+str(mapping)]
            # Get lower left corner of COSMO domain in local coordinates
            llc_COSMO = [x[0],y[0]]
            llc_COSMO = np.array(llc_COSMO).astype('float32')
            
            # Get upper left corner of COSMO domain in local coordinates
            urc_COSMO = [x[-1],y[-1]]
            urc_COSMO = np.array(urc_COSMO).astype('float32')
    
            # Get resolution             
            res_COSMO = [x[1] - x[0], y[1] - y[0]]

            # Get latitude and longitude of southern pole
            grid_mapping = data['grid_mapping_' + str(mapping)]
            lat_SP = - grid_mapping.metadata['grid_north_pole_latitude']
            lon_SP = grid_mapping.metadata['grid_north_pole_longitude'] - 180
            
            # Main loop
            global_idx = 0 # idx on radial, and all quadrature points
            vert_idx = 0 # idx only on radial and vertical quadrature points (for h profile)
            
            for radial,elev in enumerate(elevations): # Loop on all radials
                print('Processing elevations ang. '+str(elev))
                list_beams = []
                
                for j,pt_vert in enumerate(pts[1]): 
                    for i,pt_hor in enumerate(pts[0]):
                        
                        lats_radial = all_lats_radial[global_idx]
                        lons_radial = all_lons_radial[global_idx]
                        
                        # Transform radar gate coordinates into local COSMO coordinates
                        coords_rad_loc = global_to_local_coords(lons_radial,
                                             lats_radial,[lon_SP,lat_SP])  

                        # Check if all points are within COSMO domain
                        
                        if np.any(coords_rad_loc[:,0]<llc_COSMO[0]) or\
                            np.any(coords_rad_loc[:,1]<llc_COSMO[1]) or \
                            np.any(coords_rad_loc[:,0]>urc_COSMO[0]) or \
                            np.any(coords_rad_loc[:,1]>urc_COSMO[1]):
                            
                            
                            raise(IndexError('Radar domain is not entirely contained'+
                                                          ' in COSMO domain, aborting'))
                        
                        dic_beams = {}
                   
                        model_data=var[:]
                        
                        coords_rad_loc = np.fliplr(coords_rad_loc).astype('float32')
                        
                        rad_interp_values = get_all_radar_pts(len(rrange),\
                                          coords_rad_loc,all_h[vert_idx],model_data, \
                                          zlevels, llc_COSMO,res_COSMO)[1]
            
                 
                        mask_beam = np.zeros((len(rad_interp_values)))
                        # Means that the interpolated point is above COSMO domain
                        mask_beam[rad_interp_values == -9999] = -1 
                        # Means that the interpolated point is below COSMO terrain
                        mask_beam[np.isnan(rad_interp_values)] = 1
                        # Assign NaN to all missing data
                        rad_interp_values[mask_beam!=0] = np.nan 
                        dic_beams[varname] = rad_interp_values
                        
                        list_beams.append(_Beam(dic_beams, mask_beam, lats_radial,
                            lons_radial, all_s[vert_idx],all_h[vert_idx],
                            [pt_hor + azimuth,pt_vert+elev],
                            weights[i,j]))         
                    
                        # Increment global index
                        global_idx += 1
                    # Increment vertical index
                    vert_idx += 1
                    
                # Integrate all sub-beams
                scan, frac_pow = integrate_quad(list_beams)    
                
                # Add radial
                data_interp[:,radial] = scan.values[varname]
                lats_scan[:,radial] = scan.lats_profile
                lons_scan[:,radial] = scan.lons_profile
                heights_scan[:,radial] = scan.heights_profile
                dist_ground[:,radial] = scan.dist_profile
                frac_power[:,radial] = frac_pow
                
            marray = MetaArray(data_interp)
            
            # Create coordinates
            coords = OrderedDict()
            coords['rrange'] = rrange
            coords['elevation'] = elevations
            
            # Create attributes
            attrs = OrderedDict()
            attrs['heights'] = heights_scan
            attrs['lon'] = lons_scan
            attrs['lat'] = lats_scan
            attrs['azimuth'] = azimuth
            attrs['frac_power'] = frac_power
             
            # Assign coords/attrs
            marray.coordinates = coords
            marray.attributes = attrs
            
            # Assign to overall dict
            output[varname] = marray        
            
                    
    return output

if __name__ == '__main__':
    from retrieve_model_data import ssh, COSMO_query, save_netcdf,load_netcdf
    from retrieve_model_data import ELA_ADRESS,USERNAME,PASSWORD, KESCH_ADRESS
#    connection = ssh(ELA_ADRESS,USERNAME,PASSWORD)
#    connection.openChannel(KESCH_ADRESS,USERNAME)
#    query = COSMO_query(connection)
#    data = query.retrieve_data(['W','T','V','U'],'2015-04-20 08:37',coord_bounds=([10,45],[12,46]))
##
#    save_netcdf(data,'test.nc')
    import matplotlib.pyplot as plt
    data2 = load_netcdf('test.nc')
    prof = coords_profile([45.8,10.5],[45.2,11],npts=100)
    options_PPI = {'beamwidth':1.5,'azimuth':30,
           'rrange':np.arange(200,20000,1000),'npts_quad':[3,3],
           'refraction_method':1,'rpos':[11,45.5,1000]}
    ppi_U = extract(data2,['T'],'RHI',options_PPI) 
#    a = extract(data2,['T','U'],slice_type = 'lonlat',idx = prof)

    
    