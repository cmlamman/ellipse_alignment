import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Ellipse
from functions.sky_functions import *
from functions.alignment_functions import *

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)


def get_Mpc_h(deg, z=0.725):
    return deg_to_rad(deg) * (cosmo.angular_diameter_distance(z)/u.Mpc) / 0.7 # in units of Mpc/h, assuming lcdm and z=0.725
def get_deg(Mpc_h, z=0.725):
    return deg((Mpc_h* 0.7) / (cosmo.angular_diameter_distance(z)/u.Mpc))
def get_Mpc_h_comoving(deg, z=0.725):
    return deg_to_rad(deg) * (cosmo.comoving_distance(z)/u.Mpc) / 0.7 # in units of Mpc/h, assuming lcdm and z=0.725


def rw1_to_z(rw1):
    return rw1*0.2443683701202549+0.0037087929968548927



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
    
    
def plot_rel_e(results_paths=['sample_results/alignment0_'], labels=['LRG basic alignment'], 
    title='Projected Alignment of DESI LRGs', psize = 5, esize = 3, lw=1):
    
    v=100
    
    fig = plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.xlabel('Radial Separation [deg]')
    plt.ylabel(r"Relative Ellipticity of Neighbor $\epsilon_1'$")
    binx = np.linspace(0, 0.5, 20)
    
    for k in range(len(results_paths)):

        all_means_crz0 = [] 
        for r in range(100):
            try:
                dff0 = np.array(open(results_paths[k]+str(r)+'.csv').read().split('\n')[:-1]).astype(np.float)
                all_means_crz0.append(dff0)
            except FileNotFoundError:
                continue
        av_means_ab0 = np.mean(all_means_crz0, axis=0)
        av_err_ab0 = np.sqrt(np.sum((av_means_ab0-all_means_crz0)**2, axis=0)) / v # rms / sqrt(n_bins)
        all_means_ab0 = np.concatenate(all_means_crz0)
        
        plt.errorbar(binx, av_means_ab0, yerr=av_err_ab0, linestyle='--', marker='.', 
                     linewidth=lw, capsize=esize, markersize=psize, label=labels[k]);
        
        
    # more fancy things to add to plot
    
    plt.legend(loc='upper right')
    plt.plot([0, 0.5], [0, 0], color='black', linewidth=0.2, linestyle='--', dashes=(40, 30));
    
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1Xs = ax1.get_xticks()
    ax2Xs = []  
    [ax2Xs.append(str(get_Mpc_h(X, z=0.5).round())[:-2]) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(ax2Xs);
    ax2.set_xlabel('Mpc/h (z=0.5)')
    plt.tight_layout();