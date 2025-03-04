import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint

"""
Contains sets of routines to perform beam tracing for PPI and RHI data
extraction as well as some routines for antenna integration
"""

SP_COORDS_COSMO = [10,-43] # This info can be obtained from any MCH Grib file

RAD_TO_DEG = 180.0/np.pi
DEG_TO_RAD = 1./RAD_TO_DEG


class _Beam():
    def __init__(self, dic_values, mask, lats_profile, lons_profile,
                 dist_ground_profile, heights_profile, GH_pt=[], GH_weight=1):
        self.mask=mask
        self.GH_pt=GH_pt
        self.GH_weight=GH_weight
        self.lats_profile=lats_profile
        self.lons_profile=lons_profile
        self.dist_profile=dist_ground_profile
        self.heights_profile=heights_profile
        self.values=dic_values


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
    z = data['z_'+str(mapping)]

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
    N.attributes['z-levels'] = z
    return N

  
def _sum_arr(x,y, cst = 0):
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))          
            pad_1.append((0,0))
            
        
    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)
    
    z = np.sum([x,y],axis=0)
    
    return z
    
def _nansum_arr(x,y, cst = 0):

    x = np.array(x)
    y = np.array(y)
    
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))          
            pad_1.append((0,0))
        
    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)
    
    z = np.nansum([x,y],axis=0)
    return z    
    
def _piecewise_linear(x,y):
    interpolator=interp1d(x,y)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if np.isscalar(xs):
            xs=[xs]
        return np.array([pointwise(xi) for xi in xs])
        
    return ufunclike

def integrate_quad(list_GH_pts):
    n_beams = len(list_GH_pts)
    n_gates = len(list_GH_pts[0].mask)
    list_variables = list_GH_pts[0].values.keys()
 
    # Sum the mask of all beams to get overall mask
    mask = np.zeros(n_gates,) # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    frac_pow = np.zeros(n_gates,) # Fraction of power received at antenna

    for i,p in enumerate(list_GH_pts):
        mask = _sum_arr(mask,p.mask) # Get mask of every Beam
        frac_pow = _sum_arr(frac_pow, (p.mask==0).astype(float)*p.GH_weight)

    mask/=float(n_beams) #mask == 1 means that every Beam is below TOPO, 
                            #smaller than 0 that at least one Beam is above COSMO domain
    
    integrated_variables={}
    for k in list_variables:
        integrated_variables[k]=[float('nan')]
        for p in list_GH_pts:
            integrated_variables[k] = _nansum_arr(integrated_variables[k],
                                        p.values[k]*p.GH_weight)
        
        integrated_variables[k][np.logical_or(mask>1,mask<=-1)] = float('nan')
        integrated_variables[k] /= frac_pow # Normalize by valid pts
            
    # Get index of central beam
    idx_0=int(n_beams/2)

    heights_radar=list_GH_pts[idx_0].heights_profile
    distances_radar=list_GH_pts[idx_0].dist_profile
    lats=list_GH_pts[idx_0].lats_profile
    lons=list_GH_pts[idx_0].lons_profile

    integrated_beam = _Beam(integrated_variables,mask,lats, lons, distances_radar, heights_radar)
    return integrated_beam, frac_pow

    
def quad_pts_weights(beamwidth, npts_GH):
        
    # Calculate quadrature weights and points
    nh_GH = npts_GH[0]
    nv_GH = npts_GH[1]
        
    # Get GH points and weights
    sigma = beamwidth/(2*np.sqrt(2*np.log(2)))

    pts_hor, weights_hor=np.polynomial.hermite.hermgauss(nh_GH)
    pts_hor=pts_hor*sigma
   
    pts_ver, weights_ver=np.polynomial.hermite.hermgauss(nv_GH)
    pts_ver=pts_ver*sigma

    weights = np.outer(weights_hor*sigma,weights_ver*sigma)
    weights *= np.abs(np.cos(pts_ver))
    
    sum_weights = np.sum(weights.ravel())

    return [pts_hor,pts_ver], weights/sum_weights
        
def earth_radius(latitude):
    a=6378.1370*1000 # largest radius
    b=6356.7523*1000 # smallest radius
    return np.sqrt(((a**2*np.cos(latitude))**2+\
                    (b**2*np.sin(latitude))**2)/((a*np.cos(latitude))**2\
                    +(b*np.sin(latitude))**2))

def refraction_sh(range_vec, elevation_angle, coords_radar, refraction_method, N=0):
    # Method can be '4/3', 'ODE_s'
    # '4/3': Standard 4/3 refraction model (offline, very fast)
    # ODE_s: differential equation of atmospheric refraction assuming horizontal homogeneity

    if refraction_method == 1:
        S,H = ref_4_3(range_vec, elevation_angle, coords_radar)
    elif refraction_method == 2:
        S,H = ref_ODE_s(range_vec, elevation_angle, coords_radar, N)

    return S,H  
    
def deriv_z(z,r,n_h_spline, dn_dh_spline, RE):
    # Computes the derivatives (RHS) of the system of ODEs
    h,u=z
    n=n_h_spline(h)
    dn_dh=dn_dh_spline(h)
    return [u, -u**2*((1./n)*dn_dh+1./(RE+h))+((1./n)*dn_dh+1./(RE+h))]
    
def ref_ODE_s(range_vec, elevation_angle, coords_radar, N):
    
    # Get info about COSMO coordinate system
    proj_COSMO = N.attributes['proj_info']
    coords_rad_in_COSMO = global_to_local_coords(coords_radar[0], coords_radar[1],
                              [proj_COSMO['Longitude_of_southern_pole'],
                               proj_COSMO['Latitude_of_southern_pole']])
    
    llc_COSMO=(float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    res_COSMO=N.attributes['resolution']
    
    # Get position of radar in COSMO coordinates
     # Note that for lat and lon we stay with indexes but for the vertical we have real altitude s
    pos_radar_bin=[(coords_rad_in_COSMO[0][0]-llc_COSMO[0])/res_COSMO[0],
                   (coords_rad_in_COSMO[0][1]-llc_COSMO[1])/res_COSMO[1]]
                   
    # Get refractive index profile from refractivity estimated from COSMO variables
    n_vert_profile=1+(N[:,int(np.round(pos_radar_bin[0])),
                             int(np.round(pos_radar_bin[1]))])*1E-6
    # Get corresponding altitudes
    h = N.attributes['z-levels'][:,int(np.round(pos_radar_bin[0])),
                                int(np.round(pos_radar_bin[1]))] 
    h = 0.5 * (h[1:] + h[0:-1]) # h is defined on half-levels

    # Get earth radius at radar latitude
    RE = earth_radius(coords_radar[0])

    # Invert to get from ground to top of model domain
    h=h[::-1]
    n_vert_profile=n_vert_profile[::-1] # Refractivity
        
    # Create piecewise linear interpolation for n as a function of height
    n_h_spline = _piecewise_linear(h, n_vert_profile)
    dn_dh_spline = _piecewise_linear(h[0:-1],np.diff(n_vert_profile)/np.diff(h))

    z_0 = [coords_radar[2],np.sin(np.deg2rad(elevation_angle))]
    # Solve second-order ODE
    Z = odeint(deriv_z,z_0,range_vec,args=(n_h_spline,dn_dh_spline,RE))
    H = Z[:,0] # Heights above ground
    E = np.arcsin(Z[:,1]) # Elevations
    S = np.zeros(H.shape) # Arc distances
    dR = range_vec[1]-range_vec[0]
    S[0] = 0
    for i in range(1,len(S)): # Solve for arc distances
        S[i] = S[i-1]+RE*np.arcsin((np.cos(E[i-1])*dR)/(RE+H[i]))

    return S.astype('float32'), H.astype('float32')
    
def ref_4_3(range_vec, elevation_angle, coords_radar):
    elevation_angle=elevation_angle*np.pi/180.
    ke=4./3.
    altitude_radar=coords_radar[2]
    latitude_radar=coords_radar[1]
    # Compute earth radius at radar latitude 
    EarthRadius = earth_radius(latitude_radar)
    # Compute height over radar of every range_bin        
    H=np.sqrt(range_vec**2 + (ke*EarthRadius)**2+2*range_vec*ke*EarthRadius*np.sin(elevation_angle))-ke*EarthRadius+altitude_radar
    # Compute arc distance of every range bin
    S=ke*EarthRadius*np.arcsin((range_vec*np.cos(elevation_angle))/(ke*EarthRadius+H))
    
    return S.astype('float32'),H.astype('float32')


def global_to_local_coords(lon,lat,sp_coord = None, option = 1):

    if not sp_coord:
        sp_coord = SP_COORDS_COSMO
      
    lon = (lon*np.pi)/180 # Convert degrees to radians
    lat = (lat*np.pi)/180

    sp_lon = sp_coord[0]
    sp_lat = sp_coord[1]

    theta = 90 + sp_lat # Rotation around y-axis
    phi = sp_lon # Rotation around z-axis

    phi = (phi*np.pi)/180 # Convert degrees to radians
    theta = (theta*np.pi)/180

    x = np.cos(lon)*np.cos(lat) # Convert from spherical to cartesian coordinates
    y = np.sin(lon)*np.cos(lat)
    z = np.sin(lat)

    if(option == 1): # Regular -> Rotated

        x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z
        y_new = -np.sin(phi)*x + np.cos(phi)*y
        z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z

    elif(option == 2): # Rotated -> Regular

        phi = -phi
        theta = -theta

        x_new = np.cos(theta)*np.cos(phi)*x + np.sin(phi)*y + np.sin(theta)*np.cos(phi)*z
        y_new = -np.cos(theta)*np.sin(phi)*x + np.cos(phi)*y - np.sin(theta)*np.sin(phi)*z
        z_new = -np.sin(theta)*x + np.cos(theta)*z

    lon_new = np.arctan2(y_new,x_new) # Convert cartesian back to spherical coordinates
    lat_new = np.arcsin(z_new)

    lon_new = (lon_new*180)/np.pi # Convert radians back to degrees
    lat_new = (lat_new*180)/np.pi

    grid_out = np.vstack((lon_new, lat_new)).T

    return grid_out



