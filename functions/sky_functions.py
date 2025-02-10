# Useful functions for dealing with the coordinates of and making sky catalogs

from astropy import units as u
from astropy.coordinates import SkyCoord
#from astropy.coordinates.tests.utils import randomly_sample_sphere
import numpy as np

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)

# converting between deg and radians ( if not already in astropy anlge)
def rad_to_deg(ang_rad):
    return ang_rad * 180 / np.pi
def deg_to_rad(ang_deg):
    return ang_deg * np.pi / 180

def ang180(ang):
    '''Put orientation angle in range of 0-180 deg'''
    ang = ang % 180
    ang = (ang+180)%180
    return ang

def angpi(ang):
    '''Put orientation angle in range of 0-pi deg'''
    ang = ang % np.pi
    ang = (ang+np.pi)%np.pi
    return ang


def sky_area(ra1, ra2, dec1, dec2):
    return np.abs((ra2-ra1)*(np.sin(dec2*np.pi/180.)-np.sin(dec1*np.pi/180.))*(180./np.pi))
total_skyarea = sky_area(0, 360, -90, 90)


def get_points(data):
    '''ra and dec in deg, returns points on unit circle (ie distances in radians)'''
    points = SkyCoord(data['RA'], data['DEC'], unit='deg', frame='icrs', equinox='J2000.0')
    points = points.cartesian   # old astropy: points.representation = 'cartesian'
    return np.dstack([points.x.value, points.y.value, points.z.value])[0]


def get_sep(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - astropy quantity of separation 
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    return (c1.separation(c2)).to(u_result)

def get_sep_2D(catalog1, catalog2, u_coords='deg', u_result=u.deg):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
       separation in units of u_coords
    '''
    c1 = SkyCoord(catalog1['RA'], catalog1['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(catalog2['RA'], catalog2['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    return (c1.separation(c2)).to(u_result).value
    
def get_pa(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - astropy quantity of position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    pa = c1.position_angle(c2).to(u_result)
    return pa


def get_sep_pa(ra1, dec1, ra2, dec2, u_coords='deg'):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    sep = c1.separation(c2).to(u.rad)
    pa = c1.position_angle(c2).to(u.rad)
    return sep, pa



def rand_sample(catalog, n):
    return catalog[np.random.choice(len(catalog), n, replace=False)]


def rand_targets(num=100, ra1=162., ra2=198., dec1=2., dec2=28.):
    '''
    creates ra and decs [degs] for num objects inside fiducial region defined by input ra/decs
    '''
     
    area = sky_area(ra1, ra2, dec1, dec2)
    #n = int(density*total_skyarea)  
    n = int((num / area) * total_skyarea * 1.5)  # just to make sure there'll be enough in the desired region
        
    
    ra, dec, _ = randomly_sample_sphere(n)
    c = SkyCoord(ra, dec)
    coords=np.asarray([c.ra.deg, c.dec.deg]).transpose()
    
    #only take coords in region
    coords_lim = coords[((coords[:,0]>=ra1) & (coords[:,0]<=ra2) & (coords[:,1]>=dec1) & (coords[:,1]<=dec2))]
    coords_final = rand_sample(coords_lim, num)  # make final sample exactly how many was asked for
    
    return coords_final.transpose()    




## functions to limit a table of objects to only those within a given boundary

def limit_region(targets, ra1=200., ra2=205., dec1=0., dec2=5.):
    '''input targets [astropy table] and ra/dec limits'''
    try:
        return targets[(targets['RA']>ra1)&(targets['RA']<ra2)&(targets['DEC']>dec1)&(targets['DEC']<dec2)]
    except KeyError:
        try:
            return targets[(targets['TARGET_RA']>ra1)&(targets['TARGET_RA']<ra2)&(targets['TARGET_DEC']>dec1)&(targets['TARGET_DEC']<dec2)]
        except KeyError:
            return targets[(targets['RA']>ra1)&(targets['RA']<ra2)&(targets['Dec']>dec1)&(targets['Dec']<dec2)]

def radial_region(targets, ra, dec, r):
    '''limit to region within r [deg] of given coords [deg]'''
    return targets[(get_sep(ra, dec, targets['RA'], targets['DEC'])<r)]

def donut_region(targets, ra, dec, r_min, r_max):
    '''limit to region within r [deg] of given coords [deg]'''
    seps = get_sep(ra, dec, targets['RA'], targets['DEC'])
    return targets[(seps<=r_max) & (seps>r_min)]


def get_Mpc_h(deg, z=0.725):
    return deg_to_rad(deg) * (cosmo.angular_diameter_distance(z)/u.Mpc) / 0.7 # in units of Mpc/h, assuming lcdm and z=0.725
def get_deg(Mpc_h, z=0.725):
    return deg((Mpc_h* 0.7) / (cosmo.angular_diameter_distance(z)/u.Mpc))
def get_Mpc_h_comoving(deg, z=0.725):
    return deg_to_rad(deg) * (cosmo.comoving_distance(z)/u.Mpc) / 0.7 # in units of Mpc/h, assuming lcdm and z=0.725