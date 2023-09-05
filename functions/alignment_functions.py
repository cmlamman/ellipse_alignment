import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from functions.sky_functions import *

import time, glob

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)

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

def psep_to_lmi(psep):
    '''return index (0-20) corresponding to 20 projected separations [deg] for use in look-up table of weights'''
    psep[psep>0.5] = 0.499
    return (((psep)/0.5)*20).astype(int)


def get_psep(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - astropy quantity of separation 
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    return (c1.separation(c2)).to(u_result).value

def get_weight_3D(catalog1, catalog2, weights):
    '''catalogs must consitute pairs which are already within 0.5 deg of each other'''
    color1_index = rw1_to_lmi(catalog1['rw1'])
    color2_index = rw1_to_lmi(catalog2['rw1'])
    psep_difference = get_psep(catalog1['RA'], catalog1['DEC'], catalog2['RA'], catalog2['DEC'], u_result=u.deg)
    psep_index = psep_to_lmi(psep_difference)
    return weights[color1_index, color2_index, psep_index]


def get_rel_es(catalog, indices, weights=None, rcolor='rw1'):
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
        return e1_re, e2_rel, None
    
    else:
        
        rw1_centers = catalog[rcolor][ci]
        rw1_neighbors = catalog[rcolor][ni]
        all_ws = weights[rw1_to_lmi(rw1_centers), rw1_to_lmi(rw1_neighbors)]
        #all_ws = get_weight_3D(centers_m, neighbors_m, weights)**2
    
        return e1_re, e2_rel, all_ws
    
    

def get_e_dist(catalog, tree, n_centers, max_dist=deg_to_rad(0.5), max_neighbors=100, centers=None, delta_rw1_min=None, delta_rw1_max=None, rw1_positive=True, no_lensed=False, weights=None, rcolor='rw1'):
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
            too_close = (catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) < delta_rw1_min   #rw1 of neighbor - rw1 of center
        elif rw1_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_close = (catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) > -delta_rw1_min 
        elif rw1_positive == None:
            too_close == np.abs(catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) < delta_rw1_min 
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    if delta_rw1_max!=None:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        # indices where pairs are too far in r-w1 color
        drz = (catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) #rw1 of neighbor - rw1 of center
        if rw1_positive == False:  # remove where neighbor is ~in front of central galaxy
            too_far = (np.abs(drz) > delta_rw1_max) & (drz < 0)
        elif rw1_positive == None:
            # second condition temporary for test
            radial_distances =  cosmo.comoving_distance(catalog['Z']).to(u.Mpc).value
            rad_seps = np.abs(radial_distances[ii[:,:1]] - radial_distances[ii])
            too_far = (np.abs(catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) > delta_rw1_max) | (rad_seps > 60)
        catalog.remove_row(-1)
        too_far[:,:1]=False # don't want to remove indices of centers
        ii[too_far] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_far] = float('inf')
        
    if no_lensed==True:  # only pairs where neighbor is in front of center
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        too_close = (catalog[rcolor][ii[:,:1]] - catalog[rcolor][ii]) < 0  # remove if redshift of center < redshift of neighbor galaxy
        catalog.remove_row(-1)
        too_close[:,:1]=False # don't want to remove indices of centers
        ii[too_close] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[too_close] = float('inf')
        
    # measure relative ellipticities
    rel_es, rel2_es, weights_tu = get_rel_es(catalog, ii, weights=weights, rcolor=rcolor)
    
    # removing seperations where there is no neighbor
    seps = dd[:,1:].ravel()
    seps = seps[seps!= float('inf')]   # separation in radians
    
    return seps, rel_es, weights_tu



# FOR SAVING CONDENSED VERESION OF RESULTS

def bin_results(seps, reles, nbins=20, sep_max=deg_to_rad(0.5), weights=None): 
    '''sep_max really does nothing'''
    
    if(weights is not None):
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
                      rcolor='rw1', delta_rw1_min=None, delta_rw1_max=2, rw1_positive=None, 
                      sort_by='sky area', overwrite=True):
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
    
    k0=0
    if overwrite==False:
        k0 = len(glob.glob(save_path+'*.csv')) # number of batches already saved
    
    if sort_by=='sky area':
    
        if(weights is not None):
            weights0 = np.load(weights)
        elif(weights is None):
            weights0 = None

        data.sort('DEC')
        # make tree
        combined_points = get_points(data)
        tree = cKDTree(combined_points)
    
        v=10  # number of dec strips
        nn=int(len(data)/v)  # size of dec strips
        n0=0
        n1=nn
        k=0 # to keep track of number of squares
        
        # go through v batches and save each time
        for r in range(v):   # can add [n:] - starting on n if that's were it ended last time
                
            if r%2==True:
                print('Working on '+str(r+1)+'/'+str(v))
                print("So far it's been",round((time.time()-t0)/60., 10),' minutes\n')

            catalog0 = data[n0:n1].copy()
            n0+=nn; n1+=nn

            catalog0.sort('RA')
            w = 10                 # number of ra strips
            mm=int(len(catalog0)/w)  # size of squares after strips split into ra
            m0=0
            m1=mm
            for s in range(w):
                catalog = catalog0[m0:m1]
                m0+=mm; m1+=mm
                k+=1
                
                if k<=k0:
                    continue
                
                # seps in radians
                seps, rele1s, weights_tu = get_e_dist(data, tree, len(catalog), max_dist=deg_to_rad(0.5),
                                                      max_neighbors=2000, centers=catalog, weights=weights0,
                                                      delta_rw1_min=delta_rw1_min, delta_rw1_max=delta_rw1_max,
                                                      rw1_positive=rw1_positive, rcolor=rcolor) 
                # binning
                binx, wmeans, stds = bin_results(seps, rele1s, nbins=20, weights=weights_tu, sep_max=deg_to_rad(0.5))
                # in radians

                #print('Saving') 
                np.savetxt(save_path+str(k)+'.csv', wmeans, delimiter=",")

        t1 = time.time()    
        print('Finished! Total time: ',round((t1-t0)/60., 10),' minutes\n')    
        
    
    
    
    
    elif sort_by=='default order':
    
        if(weights is not None):
            weights0 = np.load(weights)
        elif(weights is None):
            weights0 = None

        # make tree
        combined_points = get_points(data)
        tree = cKDTree(combined_points)
        
        v = 100                 # number of batches to run in (data will be saved after each batch)
        nn=int(len(data)/v)    
        n0=0
        n1=nn

        # go through v batches and save each time
        for r in range(v):   # can add [n:] - starting on n if that's were it ended last time

            if r%10==True:
                print('Working on '+str(r)+'/'+str(v))
                print("So far it's been",round((time.time()-t0)/60., 3),' minutes\n')    
            catalog = data[n0:n1]
            n0+=nn; n1+=nn
            
            if (r+1)<=k0:
                    continue
                    
            seps, rele1s, weights_tu = get_e_dist(data, tree, len(catalog), max_dist=deg_to_rad(0.5),
                                                  max_neighbors=2000, centers=catalog, weights=weights0,
                                                  delta_rw1_min=delta_rw1_min, delta_rw1_max=delta_rw1_max,
                                                  rw1_positive=rw1_positive, rcolor=rcolor) 

            # binning
            binx, wmeans, stds = bin_results(seps, rele1s, nbins=20, sep_max=deg_to_rad(0.5), weights=weights_tu)

            #print('Saving') 
            np.savetxt(save_path+str(r+1)+'.csv', wmeans, delimiter=",")

        t1 = time.time()    
        print('Finished! Total time: ',round((t1-t0)/60., 10),' minutes\n')
        
        
def get_IA_weights(catalog, axis_ratio_column='axis_ratio', position_angle_column='position_angle', ra_column='RA', dec_column='Dec'):
    '''
    catalog: table where each row contains a pair of galaxies along with their:
        axis_ratio: float between 0-1
        position_angle: float, degrees between 0-180. Measured E of N
        RA, DEC: float, in degrees
    '''
    # position angle between the galaxies in each pair
    pa1 = get_pa(catalog[ra_column+'1'], catalog[dec_column+'1'], catalog[ra_column+'2'], catalog[dec_column+'2']).value
    pa2 = angpi(pa1) # equivalent to pa1 + pi
    
    a = np.asarray([1]*len(catalog)) # since we have axis ratios
    # calculate rotation angle of galaxy1 relative to seperation vector between it and galaxy2
    pa_rel1 = catalog[position_angle_column+'_1'] - pa1  # in rad
    e1_rel1, e2_rel = e_complex(a, catalog[axis_ratio_column+'_1'], pa_rel1)
    rel_ellipticity1 = np.asarray(e1_rel1)
    
    # and now do the same for the other way around
    pa_rel2 = catalog[position_angle_column+'_2'] - pa2  # in rad
    e1_rel2, e2_rel = e_complex(a, catalog[axis_ratio_column+'_2'], pa_rel2)
    rel_ellipticity2 = np.asarray(e1_rel2)
    
    return rel_ellipticity1, rel_ellipticity2