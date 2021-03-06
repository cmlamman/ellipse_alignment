import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from functions.sky_functions import *

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

def get_rel_es(catalog, indices, weights=None):
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
    
    

def get_e_dist(catalog, tree, n_centers, max_dist=deg_to_rad(0.5), max_neighbors=50, centers=None, delta_rw1_min=None, delta_rw1_max=None, rw1_positive=True, no_lensed=False, weights=None):
    '''
    Input: astropy table of galaxies, number of centers to use, 
    maximum distance away from those centers to seach (in radians)
    delta_rw1_min: minimum difference in r-w1 color to keep pair
    rw1_positive: whether (r-w1)_primary - (r-w1)_secondary > + delta_rw1_min, or < - delta_rw1_min (False)
    weights: 20x20 numpy array with indices corresponding to 20 consecutive bins of r-w1 color from 1-4.5
    ----
    Returns: 2 arr of dimmension ([number of pairs],), one for separation from a center 
    and one for ellipticity relative to the separation vector to the center (in radians)'''
    
    if centers==None:
        centers = catalog[np.random.choice(len(catalog), n_centers, replace=False)]
    center_points = get_points(centers)
    
    # query nearest neighbors
    # dd is distances, ii is indices
    dd, ii = tree.query(center_points, distance_upper_bound=max_dist, k=max_neighbors) 
    
    # for limiting to only pairs that are sufficiently separated in r-z (ie ~ redshift)
    if delta_rw1_min!=None:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        # indices where pairs are too close in r-w1 color, whether positive or negative (center is closer / further)
        if rw1_positive == True:  # remove where neighbor is ~behind central galaxy
            too_close = (catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) < delta_rw1_min   #rw1 of neighbor - rw1 of center
        elif rw1_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_close = (catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) > -delta_rw1_min 
        elif rw1_positive == None:
            too_close == np.abs(catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) < delta_rw1_min 
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    if delta_rw1_max!=None:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        # indices where pairs are too far in r-w1 color
        drz = (catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) #rw1 of neighbor - rw1 of center
        if rw1_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_far = (np.abs(drz) > delta_rw1_max) & (drz < 0)
        elif rw1_positive == None:
            too_far = np.abs(catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) > delta_rw1_max 
        catalog.remove_row(-1)
        too_far[:,:1]=False # don't want to remove indices of centers
        ii[too_far] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_far] = float('inf')
        
    if no_lensed==True:  # only pairs where neighbor is in front of center
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        too_close = (catalog['rw1'][ii[:,:1]] - catalog['rw1'][ii]) < 0  # remove if redshift of center < redshift of neighbor galaxy
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    # measure relative ellipticities
    rel_es, rel2_es, weights_tu = get_rel_es(catalog, ii, weights=weights)
    
    # removing seperations where there is no neighbor
    seps = dd[:,1:].ravel()
    seps = seps[seps!= float('inf')]
    
    return seps, rel_es, weights_tu



# FOR SAVING CONDENSED VERESION OF RESULTS

def bin_results(seps, reles, nbins=20, sep_max=deg_to_rad(0.5), weights=0): 
    
    if len(weights)>0:
        binx = np.linspace(0, sep_max, nbins)
        
        msum, edges, binnumber = stats.binned_statistic(seps, reles*weights, statistic="sum", bins=nbins)
        wsum, edges, binnumber = stats.binned_statistic(seps, weights, statistic="sum", bins=nbins)
        wmeans = msum / wsum
        
        stds, edges, binnumber = stats.binned_statistic(seps, reles, statistic="std", bins=nbins)
        
        return binx, wmeans, stds
        
    else:
        binx = np.linspace(0, sep_max, nbins)
        means, edges, binnumber = stats.binned_statistic(seps, reles, statistic="mean", bins=nbins)
        stds, edges, binnumber = stats.binned_statistic(seps, reles, statistic="std", bins=nbins)
        return binx, means, stds
    
    
def measure_alignment(data, weights='sample_data/rw1_weights.npy', save_path='sample_results/alignment0_',
                      delta_rw1_min=None, delta_rw1_max=2, rw1_positive=None, sort_by='default_order'):
    '''
    weights: lookup matrix with weights based on chance that two galaxies with r-w1 
        colors are seperated by less than 10 Mpc. Calibrated with DESI early spectra
        (path to .npy file). Can set to "None"
    sort_by: options for how to sort the data before run in batches. 
        Doesn't matter if tree is made from full catalog, as is default.
    save_path: directory and first part of filename to save results in (str)
    _ for other args see help(get_e_dist) _
    '''
    t0 = time.time()
    
    # make tree
    combined_points = get_points(data)
    tree = cKDTree(combined_points)
    
    
    
    if sort_by=='sky area':
        
        data.sort('DEC')
        strip_width = int(len(data)/10)
    
        if len(weights)>0:
            weights0 = np.load(weights)

        v=10  # number of dec strips
        nn=int(len(LRGs)/10)+1  # size of dec strips
        n0=0
        n1=nn
        k=0 # to keep track of number of squares
        
        # go through v batches and save each time
        for r in range(v):   # can add [n:] - starting on n if that's were it ended last time
            
            if r%10==True:
                print('Working on '+str(r+1)+'/'+str(v))
                print("So far it's been",round((time.time()-t0)/60., 10),' minutes\n')

            catalog0 = data[n0:n1]
            n0+=nn; n1+=nn

            catalog0.sort('DEC')
            w = 10                 # number of ra strips
            mm=int(len(catalog0)/10)+1  # size of squares after strips split into ra
            m0=0
            m1=mm
            for s in range(w):
                catalog = catalog0[m0:m1]
                m0+=mm; m1+=mm
                k+=1


            
            seps, rele1s, weights_tu = get_e_dist(data, tree, len(catalog), max_dist=deg_to_rad(0.5),
                                                  max_neighbors=1000, centers=catalog, weights=weights0,
                                                  delta_rw1_min=delta_rw1_min, delta_rw1_max=delta_rw1_max,
                                                  rw1_positive=rw1_positive) 

            # binning
            binx, wmeans, stds = bin_results(seps, rele1s, nbins=20, sep_max=0.5, weights=weights_tu)

            #print('Saving') 
            np.savetxt(save_path+str(r+1)+'.csv', wmeans, delimiter=",")

        t1 = time.time()    
        print('Finished! Total time: ',round((t1-t0)/60., 10),' minutes\n')    
        
    
    
    
    
    elif sort_by=='default_order':
    
        if len(weights)>0:
            weights0 = np.load(weights)

        v = 100                 # number of batches to run in (data will be saved after each batch)
        nn=int(len(data)/v)+1    
        n0=0
        n1=nn

        # go through v batches and save each time
        for r in range(v):   # can add [n:] - starting on n if that's were it ended last time

            if r%10==True:
                print('Working on '+str(r)+'/'+str(v))
                print("So far it's been",round((time.time()-t0)/60., 3),' minutes\n')    
            catalog = data[n0:n1]
            n0+=nn; n1+=nn

            seps, rele1s, weights_tu = get_e_dist(data, tree, len(catalog), max_dist=deg_to_rad(0.5),
                                                  max_neighbors=1000, centers=catalog, weights=weights0,
                                                  delta_rw1_min=delta_rw1_min, delta_rw1_max=delta_rw1_max,
                                                  rw1_positive=rw1_positive) 

            # binning
            binx, wmeans, stds = bin_results(seps, rele1s, nbins=20, sep_max=0.5, weights=weights_tu)

            #print('Saving') 
            np.savetxt(save_path+str(r+1)+'.csv', wmeans, delimiter=",")

        t1 = time.time()    
        print('Finished! Total time: ',round((t1-t0)/60., 10),' minutes\n')