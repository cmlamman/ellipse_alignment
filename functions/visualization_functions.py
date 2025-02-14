import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Ellipse
from functions.sky_functions import *
from functions.alignment_functions import *
import numpy.linalg as linalg

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



def plot_E_flat(ellipse_as, ellipse_bs, thetas, color='blue', scale=1, figsize=(8,8)):
    '''for plotting single ellipse centered at orgin. thetas must be in deg'''
    
    ells = [Ellipse(xy = [0,0],
                    width=ellipse_as*scale, 
                    height=ellipse_bs*scale,
                    angle=90-thetas)]
    
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=figsize)
    ax.add_artist(ells[0])
    ells[0].set_alpha(.3)
    

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
    
    
def plot_ellipsoid_grid(el, proj_direction='x', centers=False, size=1, point_size=0.02):
    # ellispsoid and center in matrix form
    A = np.array([el['sigman_L2com'][0]**2*el['sigman_eigenvecsMaj_L2com'],
                            el['sigman_L2com'][1]**2*el['sigman_eigenvecsMid_L2com'],
                            el['sigman_L2com'][2]**2*el['sigman_eigenvecsMin_L2com']])
    
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    scale = (1/el['sigman_L2com'][0])*size
    radii = np.sqrt(s) * scale
    
    # making 3d grid around surface of ellipsoid
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # rotate grid 
    for i in range(len(x)):
        for j in range(len(v)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation)
            
    if centers=='Sky':
        y+=el['RA']; z+=el['DEC']
    if centers==True:
        x+=el['x_L2com'][0]; y+=el['x_L2com'][1]; z+=el['x_L2com'][2]
        
    if proj_direction=='z':
        plt.scatter(x, y, color='orange', s=point_size, alpha=.5)
    if proj_direction=='y':
        plt.scatter(x, z, color='orange', s=point_size, alpha=.5)
    if proj_direction=='x':
        plt.scatter(y, z, color='orange', s=point_size, alpha=.5)
    
    
    
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
    
    
def plot_rel_e_phot(results_paths=['sample_results/alignment0_'], labels=['LRG basic alignment'], 
    title='Projected Alignment of DESI LRGs', psize = 5, esize = 3, lw=1, e_sep=False):
    '''e_sep: plot relative ellipticity x separation'''
    
    v=100
    
    fig = plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.xlabel('Radial Separation [deg]')
    if e_sep==False:
        plt.ylabel(r"Relative Ellipticity of Neighbor $\epsilon_1'$")
    elif e_sep==True:
        plt.ylabel(r"$\epsilon_1'$(separation)")
    binx = np.linspace(0, 0.5, 20)
    
    for k in range(len(results_paths)):

        all_means_crz0 = [] 
        for r in range(100):
            try:
                dff0 = np.array(open(results_paths[k]+str(r)+'.csv').read().split('\n')[:-1][1].split(',')).astype(float)
                all_means_crz0.append(dff0)
            except FileNotFoundError:
                continue
        av_means_ab0 = np.mean(all_means_crz0, axis=0)
        av_err_ab0 = np.sqrt(np.sum((av_means_ab0-all_means_crz0)**2, axis=0)) / np.sqrt(v*(v-1))  # rms / sqrt(n_bins)
        all_means_ab0 = np.concatenate(all_means_crz0)
        
        if e_sep==False:
            plt.errorbar(binx, av_means_ab0, yerr=av_err_ab0, linestyle='--', marker='.', 
                         linewidth=lw, capsize=esize, markersize=psize, label=labels[k]);
        elif e_sep==True:
            sep_err_frac_est = .01
            errs = np.abs(av_means_ab0*binx) * np.sqrt((av_err_ab0/av_means_ab0)**2 + (.01)**2)
            plt.errorbar(binx, av_means_ab0*binx, yerr=errs, linestyle='--', marker='.', 
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
    
    
    
def plot_rel_e(results_paths=['sample_results/alignment0_'], labels=['LRG basic alignment'], colors=['b'],
    title=' ', psize = 5, esize = 3, lw=1, e_sep=False, Ls=[[1]], alpha=1, save_path=False):
    '''e_sep: plot relative ellipticity x separation'''
    lbs=15; fts=17
    #v= 100
    
    if Ls[0][0]==1:
        Ls = [[1]]*len(results_paths)
    
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    #ax1.ticklabel_format(style='sci', axis='y')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(title)
    plt.xlabel(r'Projected Separation, $r_p$ [$h^{-1}$Mpc]', fontsize=fts)
    
    if e_sep==False and Ls[0][0]==1:
        plt.ylabel(r"Relative Ellipticity of Neighbor $\mathcal{E} (r_p)$", fontsize=fts)
    elif e_sep==True and Ls[0][0]==1:
        plt.ylabel(r"$R\mathcal{E} (r_p)$ [$h^{-1}$Mpc]", fontsize=fts)
    
    elif e_sep==False and Ls[0][0]!=1:
        plt.ylabel(r"$L\mathcal{E} (r_p)$ [$h^{-1}$Mpc]", fontsize=fts)
    elif e_sep==True and Ls[0][0]!=1:
        plt.ylabel(r"$LR\mathcal{E} (r_p)$ [$h^{-2}$Mpc$^2$]", fontsize=fts)
    
    for k in range(len(results_paths)):

        all_means_crz0 = [] 
        for r in range(100+1)[1:]:
            try:
                file_contents = np.array([fc.split(',') for fc in open(results_paths[k]+str(r)+'.csv').read().split('\n')])
                binx = np.asarray(file_contents[0]).astype('float')
                dff0 = np.asarray(file_contents[1]).astype('float')
                all_means_crz0.append(dff0)
            except FileNotFoundError:
                continue
        v = len(all_means_crz0)
        print(v, 'files found')
        av_means_ab0 = np.mean(all_means_crz0, axis=0)
        av_err_ab0 = np.sqrt(np.sum((av_means_ab0-all_means_crz0)**2, axis=0)) / np.sqrt(v*(v-1))  # rms / sqrt(n_bins)
        all_means_ab0 = np.concatenate(all_means_crz0)
        
        if e_sep==False:
            plt.errorbar(binx, av_means_ab0*Ls[k], yerr=av_err_ab0*Ls[k], linestyle='--', marker='.', 
                         linewidth=lw, capsize=esize, markersize=psize, label=labels[k], color=colors[k], alpha=alpha);
            plt.legend(loc='upper right', fontsize=lbs)
            
            
        elif e_sep==True:
            sep_err_frac_est = .01
            errs = np.abs(av_means_ab0*binx) * np.sqrt((av_err_ab0/av_means_ab0)**2 + (.01)**2)
            plt.errorbar(binx, av_means_ab0*binx*Ls[k], yerr=errs*Ls[k], linestyle='--', marker='.', 
                         linewidth=lw, capsize=esize, markersize=psize, label=labels[k], color=colors[k], alpha=alpha);
            legend = plt.legend(loc='upper left', fontsize=lbs)#, title='Density tracer')
            plt.setp(legend.get_title(),fontsize='large')
    
    plt.plot([0, np.max(binx)], [0, 0], color='black', linewidth=0.2, linestyle='--', dashes=(40, 30));
    
    plt.xticks(fontsize=lbs); plt.yticks(fontsize=lbs);
    
    if save_path!=False:
        fig.savefig(save_path, dpi=300)