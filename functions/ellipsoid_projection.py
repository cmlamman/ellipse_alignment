import numpy as np
from astropy.table import Table

def format_ellipsoid(eigenvectors, eigenvalues, position = np.asarray([0,0,0])):
    '''
    Formatt ellipsoid parameters to match Abacus. 
    Eigenvectors and values must be in order of least to greatest
    '''
    el = Table()
    el['sigman_eigenvecsMin_L2com'] = eigenvectors[0]
    el['sigman_eigenvecsMid_L2com'] = eigenvectors[1]
    el['sigman_eigenvecsMaj_L2com'] = eigenvectors[2]
    
    el['sigman_L2com'] = np.sqrt(eigenvalues)
    el['sigma_L2com'] = position
    
    return el


def get_projected_shape_sphere(el):
    '''project abacus ellipse onto y-z plane. returns major axis length, minor axis length, and orientation angle in radians'''
    
    # getting rotation vectors
    rot_z = R.from_rotvec(np.asarray([0, 0, -rad(el['RA'])]))
    rot_y = R.from_rotvec(np.asarray([0, rad(el['DEC']), 0]))
        
    # eigen vectors and eigen values of triaxial ellipsoid. rotate them so x lies along line of sight
    evc0 = np.array([el['sigman_eigenvecsMaj_L2com'], el['sigman_eigenvecsMid_L2com'], el['sigman_eigenvecsMin_L2com']])
    evc = np.matmul(rot_y.as_matrix(), np.matmul(rot_z.as_matrix(), evc0.transpose())).transpose() 
    
    evl = el['sigman_L2com']**2 # eigen values
    
    K = np.sum(evc[:,0][:,None] * (evc / evl[:,None]), axis=0)
    
    r = evc[:,2] - evc[:,0]*K[2]/K[0]
    s = evc[:,1] - evc[:,0]*K[1]/K[0]
    A = np.sum(r**2 / evl, axis=0)
    B = np.sum(2*r*s / evl, axis=0)
    C = np.sum(s**2 / evl, axis=0)
    
    theta = np.pi/2 + np.arctan2(B, A-C) / 2   # measured in direction of +RA from N, in radians
    b_p = 1 / np.sqrt(((A+C)/2) + ((A-C)/(2*np.cos(2*theta))))
    a_p = 1 / np.sqrt(A + C - (1/b_p**2))
    return a_p, b_p, theta



def get_projected_shape_x(el):
    '''
    project 3D ellipse onto y-z plane
    Input: ellipse with eigenvecors and eigen values, in formatt used by Abacus Summit
    https://abacussummit.readthedocs.io/en/latest/data-products.html#halo-statistics
    '''
    # eigen vectors and eigen values of triaxial ellipsoid
    evc = np.array([el['sigman_eigenvecsMaj_L2com'], el['sigman_eigenvecsMid_L2com'], el['sigman_eigenvecsMin_L2com']])
    evl = el['sigman_L2com']**2
    
    K = np.sum((evc.transpose() * evc[:,0]).transpose() / evl[:,None], axis=0)
    
    r = evc[:,2] - evc[:,0]*K[2]/K[0]
    s = evc[:,1] - evc[:,0]*K[1]/K[0]
    A = np.sum(r**2 / evl, axis=0)
    B = np.sum(2*r*s / evl, axis=0)
    C = np.sum(s**2 / evl, axis=0)
    
    theta = np.pi/2 + np.arctan2(B, A-C) / 2   # measured in direction of +RA from N, in radians
    a_p = 1 / np.sqrt(((A+C)/2) + ((A-C)/(2*np.cos(2*theta))))
    b_p = 1 / np.sqrt(A + C - (1/a_p**2))
    return a_p, b_p, theta
    
    