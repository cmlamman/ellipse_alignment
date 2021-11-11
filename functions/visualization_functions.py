import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Ellipse
from sky_functions import *
from alignment_functions import *

def plot_Es(sample, color='blue', scale=1, figsize=(8,8)):
    
    ra_decs = [z for z in zip(sample['RA'], sample['DEC'])]
    ellipse_as, ellipse_bs = a_b(sample['E1'], sample['E2'])
    thetas = rad_to_deg(get_galaxy_orientation_angle(sample['E1'], sample['E2']))
    
    ells = [Ellipse(xy = ra_decs[i],
                    width=ellipse_as[i]*scale, 
                    height=ellipse_bs[i]*scale,
                    angle=90-thetas[i]) for i in range(len(sample))]
    
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=figsize)
    for e in ells:
        ax.add_artist(e)
        e.set_alpha(.3)

    plt.xlabel('RA')
    plt.ylabel('DEC')
    ax.set_xlim(np.min(sample['RA']), np.max(sample['RA']))
    ax.set_ylim(np.min(sample['DEC']), np.max(sample['DEC']))
    
    
    
def plot_ellipces(inds, xmin=199.5, xmax=205.5, ymin=-.5, ymax=5.5, color=None):
    ells = [Ellipse(ra_decs[i],
                    width=ellipse_as[i]/4, 
                    height=ellipse_bs[i]/4,
                    angle=90-thetas[i], facecolor=colors_plot[i])
            for i in inds]

    #fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ells:
        ax.add_artist(e)
        e.set_alpha(.3)
        if color!= None:
            e.set_facecolor(color)

    plt.xlabel('RA')
    plt.ylabel('DEC')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)