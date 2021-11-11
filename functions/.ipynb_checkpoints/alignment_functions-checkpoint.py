import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

import time

from scipy.spatial import cKDTree
from scipy import stats
# useful info at http://legacysurvey.org/dr8/catalogs/

'''
ALL ANGLES, SEPARATIONS, IN RADIANS.
The only exception is RA/DEC - default is deg
'''


def get_galaxy_orientation_angle(e1, e2):
    '''return orientation angle of galaxy (range of 0-pi)'''
    return 0.5 * np.arctan2(e2, e1)

def abs_e(e1, e2):
    '''absolute value of complex ellipticity'''
    return np.sqrt(e1*e1 + e2*e2)

def e_complex(a, b, theta):
    '''complex ellipticity, theta must be in rad'''
    abs_e = (1 - (b/a)) / (1 + (b/a))
    e1 = abs_e * np.cos(2*theta)
    e2 = abs_e * np.sin(2*theta)
    return e1, e2

def a_b(e1, e2):
    '''return a and b of ellipse'''
    e = abs_e(e1, e2)
    return 1+e, 1-e  
    
       
    
# FUNCTIONS FOR MEASURING ALIGNMENT 


# METHOD 1

def rw1_to_lmi(rw1):
    '''return index (0-20) corresponding to 20 r-w1 color bins for use in look-up table of weights'''
    return (((rw1-1)/3.5)*20).astype(int)

def get_rel_es(catalog, indices, shape='default', weights=None):
    '''
    input: 
        array of indices for n centers and maximum m neighbors each- shape(n,m)
        corresponding to place in catalog
        first element of each row is indic of center
        shape can be 'ser', 'dev', or 'exp' for fit used to get ellipticity components
    returns: 
        array of same shape, containing ellipticities relative to separation
        vector between given neighbor and it's central galaxy
    '''
    
    # indices in catalog of centers and neighbors, arranges so each array is same shape
    ci = np.repeat(indices[:,0], (len(indices[0])-1)).ravel() # indices of centers
    ni = indices[:,1:].ravel()   # indices of neighbors
    # removing places where no neighbor was found in the tree
    neighbor_exists = (ni!=len(catalog))
    ci = ci[neighbor_exists]; ni = ni[neighbor_exists]
    
    centers_m = catalog[ci]
    neighbors_m = catalog[ni]   # excluding the centers
    
    # get position angle
    pa = get_pa(centers_m['RA'], centers_m['DEC'], neighbors_m['RA'], neighbors_m['DEC'])

    
    # calculate rotation angle of neighbor relative to the separation vector
    theta_neighbor = get_galaxy_orientation_angle(neighbors_m['E1'], neighbors_m['E2'])
    a, b = a_b(neighbors_m['E1'], neighbors_m['E2'])
    
    pa_rel = theta_neighbor - pa.value  # in rad
    e1_re, e2_rel = e_complex(a, b, pa_rel)
    
    if weights is None:
        return e1_re, e2_rel, 0
    
    else:
        
        rw1_centers = catalog['rw1'][ci]
        rw1_neighbors = catalog['rw1'][ni]
        all_ws = weights[rw1_to_lmi(rw1_centers), rw1_to_lmi(rw1_neighbors)]
    
        return e1_re, e2_rel, all_ws
    
    

def get_e_dist(catalog, n_centers, max_dist=deg_to_rad(0.5), max_neighbors=50, centers=None, delta_rz_min=None, delta_rz_max=None, rz_positive=True, no_lensed=False, weights=None):
    '''
    Input: astropy table of galaxies, number of centers to use, 
    maximum distance away from those centers to seach (in radians)
    delta_rz_min: minimum difference in r-z color to keep pair
    rz_positive: whether (r-z)_primary - (r-z)_secondary > + delta_rz_min, or < - delta_rz_min (False)
    weights: 20x20 numpy array with indices corresponding to 20 consecutive bins of r-w1 color from 1-4.5
    ----
    Returns: 2 arr of dimmension ([number of pairs],), one for separation from a center 
    and one for ellipticity relative to the separation vector to the center (in radians)'''
    
    if centers==None:
        centers = catalog[np.random.choice(len(catalog), n_centers, replace=False)]
    center_points = get_points(centers)
    
    # make tree
    combined_points = get_points(catalog)
    tree = cKDTree(combined_points)
    
    # query nearest neighbors
    # dd is distances, ii is indices
    dd, ii = tree.query(center_points, distance_upper_bound=max_dist, k=max_neighbors) 
    
    # for limiting to only pairs that are sufficiently separated in r-z (ie ~ redshift)
    if delta_rz_min!=None:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        # indices where pairs are too close in r-z color, whether positive or negative (center is closer / further)
        if rz_positive == True:  # remove where neighbor is ~behind central galaxy
            too_close = (catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) < delta_rz_min   #rz of neighbor - rz of center
        elif rz_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_close = (catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) > -delta_rz_min 
        elif rz_positive == None:
            too_close == np.abs(catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) < delta_rz_min 
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    if delta_rz_max!=None:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        # indices where pairs are too far in r-z color
        drz = (catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) #rz of neighbor - rz of center
        if rz_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_far = (np.abs(drz) > delta_rz_max) & (drz < 0)
        elif rz_positive == None:
            too_far = np.abs(catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) > delta_rz_max 
        catalog.remove_row(-1)
        too_far[:,:1]=False # don't want to remove indices of centers
        ii[too_far] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_far] = float('inf')
        
    if no_lensed==True:  # only pairs where neighbor is in front of center
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        too_close = (catalog['rz'][ii[:,:1]] - catalog['rz'][ii]) < 0  # remove if redshift of center < redshift of neighbor galaxy
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    # measure relative ellipticities
    rel_es, rel2_es, weights_tu = get_rel_es(catalog, ii, shape=shape, weights=weights)
    
    # removing seperations where there is no neighbor
    seps = dd[:,1:].ravel()
    seps = seps[seps!= float('inf')]
    
    if weights is None:
        return seps, rel_es
    else:
        return seps, rel_es, weights_tu



# FOR SAVING CONDENSED VERESION OF RESULTS

def bin_results(seps, reles, nbins=20, sep_max=deg_to_rad(0.5), weights=None): 
    
    if weights!=None:
        binx = np.linspace(0, sep_max, nbins)
        
        msum, edges, binnumber = stats.binned_statistic(seps, reles*weights, statistic="sum", bins=nbins)
        wsum, edges, binnumber = stats.binned_statistic(seps, weights, statistic="sum", bins=nbins)
        wmeans = msum / wsum
        
        stds, edges, binnumber = stats.binned_statistic(seps, wmeans, statistic="std", bins=nbins)
        
        return binx, wmeans, stds
        
    else:
        binx = np.linspace(0, sep_max, nbins)
        means, edges, binnumber = stats.binned_statistic(seps, reles, statistic="mean", bins=nbins)
        stds, edges, binnumber = stats.binned_statistic(seps, reles, statistic="std", bins=nbins)
        return binx, means, stds