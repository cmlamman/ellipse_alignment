import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from visualization_functions import *
from ellipsoid_projection import *
from alignment_functions import *
from lightProfile_functions import *

import time, random, scipy
from scipy.spatial.transform import Rotation as R
import scipy.special
import scipy.interpolate


def get_projected_shape_fromParameters(c, b, a=1, r_matrix='random', p_axis='x'):
    '''project 3D ellipse to random orientation given minor, mid, and major axis lengths. 
    Returns 2D axis ratio projected along p_axis'''
    
    if type(r_matrix)==str:
        r_matrix = R.random()
    r_matrix.as_matrix()

    # eigen vectors
    evc0 = np.asarray([[0,0,1],[0,1,0],[1,0,0]])
    evc = r_matrix.apply(evc0)

    evl = np.asarray([c, b, a])**2
    
    if p_axis=='x':
        K = np.sum(evc[:,0][:,None] * (evc / evl[:,None]), axis=0)
        r = evc[:,2] - evc[:,0]*K[2]/K[0]
        s = evc[:,1] - evc[:,0]*K[1]/K[0]
        
    if p_axis=='y':
        K = np.sum(evc[:,1][:,None] * (evc / evl[:,None]), axis=0)

        r = evc[:,0] - evc[:,1]*K[0]/K[1]
        s = evc[:,2] - evc[:,1]*K[2]/K[1]
    
    A = np.sum(r**2 / evl, axis=0)
    B = np.sum(2*r*s / evl, axis=0)
    C = np.sum(s**2 / evl, axis=0)
    # for p_axis='x', theta is the angle relative to z, in the direciton of +y
    # for p_axis='y', theta is the angle relative to x, in the direciton of +z
    theta = np.pi/2 + np.arctan2(B, A-C) / 2
    a_p = 1 / np.sqrt((((A+C)/2) + ((A-C)/(2*np.cos(2*theta)))))
    b_p = 1 / np.sqrt(A + C - (1/a_p**2))
    
    return (b_p/a_p), theta



### LIGHT PROFILE PARAMETERS ####################
# From https://github.com/dstndstn/tractor/blob/main/tractor/mixture_profiles.py
mix_A_luv = np.array([4.26347652e-02,   2.40127183e-01,   6.85907632e-01,   1.51937350e+00,
                    2.83627243e+00,   4.46467501e+00,   5.72440830e+00,   5.60989349e+00])
mix_r_luv = np.sqrt(np.array([2.23759216e-04,   1.00220099e-03,   4.18731126e-03,   1.69432589e-02,
                    6.84850479e-02,   2.87207080e-01,   1.33320254e+00,   8.40215071e+00]))
dev_core = 0.010233
mix_A_luv *= (1. - dev_core) / np.sum(mix_A_luv)
mix_A_luv = np.append(mix_A_luv, dev_core)
mix_r_luv = np.append(mix_r_luv, 1.0e-08)

def luv_profile(r):
    profile = r*0.0
    for j in range(len(mix_r_luv)):
        profile = profile + np.exp(-(r/mix_r_luv[j])**2/2.0)*mix_A_luv[j]/mix_r_luv[j]**3
    return profile

mix_A_lux = np.array([2.34853813e-03,   3.07995260e-02,   2.23364214e-01,
                    1.17949102e+00,   4.33873750e+00,   5.99820770e+00])
mix_r_lux = np.sqrt(np.array([1.20078965e-03,   8.84526493e-03,   3.91463084e-02,
                    1.39976817e-01,   4.60962500e-01,   1.50159566e+00]))

def lux_profile(r):
    profile = r*0.0
    for j in range(len(mix_r_lux)):
        profile = profile + np.exp(-(r/mix_r_lux[j])**2/2.0)*mix_A_lux[j]/mix_r_lux[j]**3
    return profile


def exp_profile(r):
    return scipy.special.kn(0,r)

def Hern_profile(r):
    return 1.0/r/(r+1)**3

def make_profile_rs(f, binwidth=np.full(20*100, 0.01)):
    # This is written so that one can use non-uniform bin widths in later application
    rcen = np.cumsum(binwidth)-binwidth[0]/2.0
    mass_profile = np.insert(np.cumsum(rcen**2*f(rcen)*binwidth+1e-10), 0, 0.0)
    mass_profile = mass_profile/mass_profile[-1]
    r_profile = np.insert(rcen+0.5*binwidth, 0, 0.0)
    spline = scipy.interpolate.InterpolatedUnivariateSpline(mass_profile, r_profile)
    halfmass = spline(0.5)
    rad_3d = spline((np.arange(1e5)+0.5)/1e5)/halfmass
    return rad_3d
#############################


def get_light_profile(model_type, N=100000):
    
    if model_type == 'DEV':
        r = make_profile_rs(luv_profile)
    elif model_type == 'EXP' or model_type == 'PSF' or model_type == 'REX':
        r = make_profile_rs(lux_profile)
    elif model_type == 'SER':
        r = make_profile_rs(Hern_profile)
    else:
        print(model_type)
        r = make_profile_rs(Hern_profile)
    
    r_max=1
    # generating uniform points in cube
    q=int((N*2)**(1/3))
    x_ = np.linspace(-r_max, r_max, q)
    y_ = np.linspace(-r_max, r_max, q)
    z_ = np.linspace(-r_max, r_max, q)
    x0, y0, z0 = np.meshgrid(x_, y_, z_)
    frame0 = np.asarray([x0.flatten(), y0.flatten(), z0.flatten()])
    
    dists = frame0[0]**2 + frame0[1]**2 + frame0[2]**2
    inside_r = frame0[:,(((dists)<r_max)&(dists>0))]    # reject all outside a  sphere and any that might be at 0
    h_points = inside_r[:,np.random.randint(0, len(inside_r[0]), N)]         # match number of r
    h_points = r * h_points / np.sqrt(h_points[0]**2 + h_points[1]**2 + h_points[2]**2)       # scale by radii
    return h_points



def shape_transformation_2D(a, b, c, r_matrix, h_points, scale=1, p_axis='z'):
    '''
    a, b, c: relative sizes of axis ratios
    scale: size of galaxy
    orientation_angle0, orientation_angle1: orientation about z and y, respectively
    '''
    r_matrix.as_matrix()
    scale_matrix = np.asarray([[a,0,0],[0,b,0],[0,0,c]])
    shape_scaled = np.matmul(scale_matrix, h_points)
    M3D = r_matrix.apply(shape_scaled.transpose()).transpose()
    
    if p_axis=='z':
        return M3D[:2]  # retain just the x and y direction
    if p_axis=='x':
        return M3D[1:]  # retain just the y and z direction
    



def find_r_half(h_points):
    return np.sqrt(np.median(h_points[0]**2 + h_points[1]**2))


def app_sum(M_points_2D, rad_image_sq):
    in_aperature = (M_points_2D[0]**2 + M_points_2D[1]**2) < rad_image_sq
    
    return len(M_points_2D[:,in_aperature][0])