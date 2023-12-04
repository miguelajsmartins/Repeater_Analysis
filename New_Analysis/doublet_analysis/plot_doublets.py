import pandas as pd
import numpy as np
import healpy as hp

from healpy.newvisufunc import projview, newprojplot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from scipy.stats import poisson

from datetime import datetime

import os
import sys

sys.path.append('./src/')

from hist_manip import data_2_binned_errorbar

# import event_manip
#
# from event_manip import compute_directional_exposure
# from event_manip import time_ordered_events
# from event_manip import ang_diff
# from event_manip import get_normalized_exposure_map
# from event_manip import get_skymap
# from event_manip import colat_to_dec
# from event_manip import dec_to_colat

from axis_style import set_style
#
# from array_manip import unsorted_search
#
# from compute_lambda_pvalue_binned_sky import compute_p_value

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#convert alpha into phi for ploting
def ra_to_phi(ra):
    return np.where(ra > np.pi, 2*np.pi - ra, -ra)

#round to .5 above
def round_down(x):
    return x - (x % .5)

#round to .5 below
def round_up(x):
    return x + (x % .5)

#save the number of doublets in each sample
def get_doublet_list(filelist):

    #initialize doublet list
    doublet_list = [len(pd.read_parquet(file, engine = 'fastparquet').index) for file in filelist]

    return np.array(doublet_list)

#----------------------------------------------------
# main function
#----------------------------------------------------

if __name__ == '__main__':

    #define the input path and the output path
    input_path = './datasets/iso_doublets/'
    output_path = './results'

    #initialize lists to hold files with doublet data
    files_binned_sky = []
    files_binned_targets = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if not os.path.isfile(filename):
            continue

        if 'binnedSky' in filename:
            files_binned_sky.append(filename)
        if 'binnedTargetCenters' in filename:
            files_binned_targets.append(filename)

    files_binned_sky = np.sort(files_binned_sky)
    files_binned_targets = np.sort(files_binned_targets)

    #save values of important constants
    lat_pao = np.radians(-35.15)
    theta_max = np.radians(80)
    dec_max = lat_pao + theta_max

    nside = files_binned_sky[0].split('_')[4]
    target_radius = files_binned_targets[0].split('_')[4][:3]

    #--------------------------------------
    # plot the distribution of doublets for 1 sample of each type of counting
    #--------------------------------------
    #save a colormap
    colormap = plt.get_cmap('RdBu_r')

    #define figures
    fig_doublet_skymap = plt.figure(figsize=(10, 4))

    ax_doublet_binned_sky_skymap = fig_doublet_skymap.add_subplot(121, projection='hammer')
    ax_doublet_binned_targets_skymap = fig_doublet_skymap.add_subplot(122, projection='hammer')

    #get the doublets from a given sky sample
    doublet_data_binned_sky = pd.read_parquet(files_binned_sky[0], engine = 'fastparquet')
    doublet_data_binned_targets = pd.read_parquet(files_binned_targets[0], engine = 'fastparquet')

    #save the times of the events to color the corresponding markers
    begin_time = Time('2010-01-01T00:00:00', scale = 'utc', format = 'fits').gps
    end_time = Time('2020-01-01T00:00:00', scale = 'utc', format = 'fits').gps

    doublet_color_1_binned_sky = np.interp(doublet_data_binned_sky['gps_time_1'], (doublet_data_binned_sky['gps_time_1'].min(), doublet_data_binned_sky['gps_time_1'].max()), (0, 1))
    doublet_color_2_binned_sky = np.interp(doublet_data_binned_sky['gps_time_2'], (doublet_data_binned_sky['gps_time_2'].min(), doublet_data_binned_sky['gps_time_2'].max()), (0, 1))
    doublet_color_1_binned_targets = np.interp(doublet_data_binned_targets['gps_time_1'], (doublet_data_binned_targets['gps_time_1'].min(), doublet_data_binned_targets['gps_time_1'].max()), (0, 1))
    doublet_color_2_binned_targets = np.interp(doublet_data_binned_targets['gps_time_2'], (doublet_data_binned_targets['gps_time_2'].min(), doublet_data_binned_targets['gps_time_2'].max()), (0, 1))

    #plot the doublets
    ax_doublet_binned_sky_skymap.scatter(ra_to_phi(np.radians(doublet_data_binned_sky['ra_1'])), np.radians(doublet_data_binned_sky['dec_1']), marker='o', facecolor=colormap(doublet_color_1_binned_sky), s = 5, edgecolor = 'tab:grey', linewidth = .1)
    ax_doublet_binned_sky_skymap.scatter(ra_to_phi(np.radians(doublet_data_binned_sky['ra_2'])), np.radians(doublet_data_binned_sky['dec_2']), marker='o', color=colormap(doublet_color_2_binned_sky),  s = 5, edgecolor = 'tab:grey', linewidth = .1)

    ax_doublet_binned_targets_skymap.scatter(ra_to_phi(np.radians(doublet_data_binned_targets['ra_1'])), np.radians(doublet_data_binned_targets['dec_1']), marker='o', facecolor=colormap(doublet_color_1_binned_targets), s = 5, edgecolor = 'tab:grey', linewidth = .1)
    ax_doublet_binned_targets_skymap.scatter(ra_to_phi(np.radians(doublet_data_binned_targets['ra_2'])), np.radians(doublet_data_binned_targets['dec_2']), marker='o', color=colormap(doublet_color_2_binned_targets),  s = 5, edgecolor = 'tab:grey', linewidth = .1)

    #define the style of axis
    title_doublet_binned_sky = r'Binned sky: $n_{\mathrm{side}} = %s$, $\tau = 1$ day, $N_{\tau} = %i$' % (nside, len(doublet_data_binned_sky.index))
    title_doublet_binned_targets = r'Targets in binned sky: $\psi = %s^\circ$, $\tau = 1$ day, $N_{\tau} = %i$' % (target_radius, len(doublet_data_binned_targets.index))

    ax_doublet_binned_sky_skymap = set_style(ax_doublet_binned_sky_skymap, title_doublet_binned_sky, r'$\alpha$', r'$\delta$', 14)
    ax_doublet_binned_targets_skymap = set_style(ax_doublet_binned_targets_skymap, title_doublet_binned_targets, r'$\alpha$', r'$\delta$', 14)

    ax_doublet_binned_sky_skymap.set_title(title_doublet_binned_sky, y=1.05, fontsize=14)
    ax_doublet_binned_targets_skymap.set_title(title_doublet_binned_targets, y=1.05, fontsize=14)

    #define the grip spacing
    dec_grid = np.radians(np.linspace(-60, 60, 5))
    ra_grid = np.radians(np.linspace(-150, 150, 6))

    ax_doublet_binned_sky_skymap.set_xticks(ra_grid)
    ax_doublet_binned_sky_skymap.set_yticks(dec_grid)
    ax_doublet_binned_targets_skymap.set_xticks(ra_grid)
    ax_doublet_binned_targets_skymap.set_yticks(dec_grid)

    ax_doublet_binned_sky_skymap.grid(axis='both', which='major', alpha=1)
    ax_doublet_binned_targets_skymap.grid(axis='both', which='major', alpha=1)

    #plot shadded are in skymap
    ax_doublet_binned_sky_skymap.fill_betweenx(np.linspace(dec_max, .5*np.pi , 100), -np.pi, np.pi, color='tab:grey')
    ax_doublet_binned_targets_skymap.fill_betweenx(np.linspace(dec_max, .5*np.pi , 100), -np.pi, np.pi, color='tab:grey')

    #plot color bar
    cb_doublet_binned_sky = fig_doublet_skymap.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=begin_time, vmax=end_time), cmap=colormap), ax= ax_doublet_binned_sky_skymap, orientation='horizontal')
    cb_doublet_binned_sky.ax.set_xlabel(r'$t$ (GPS second)', fontsize=14)
    cb_doublet_binned_sky.ax.tick_params(labelsize=14)

    cb_doublet_binned_targets = fig_doublet_skymap.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=begin_time, vmax=end_time), cmap=colormap), ax= ax_doublet_binned_targets_skymap, orientation='horizontal')
    cb_doublet_binned_targets.ax.set_xlabel(r'$t$ (GPS second)', fontsize=14)
    cb_doublet_binned_targets.ax.tick_params(labelsize=14)

    fig_doublet_skymap.tight_layout()
    fig_doublet_skymap.savefig(os.path.join(output_path, 'Skymap_doublets_spacetimeDist_isotropy.pdf'), dpi=1000)

    #-------------------------------------------------------------
    # plot the distribution of the number of doublets
    #-------------------------------------------------------------
    fig_doublet_dist = plt.figure(figsize=(10, 4))

    ax_doublet_binned_sky_dist = fig_doublet_dist.add_subplot(121)
    ax_doublet_binned_targets_dist = fig_doublet_dist.add_subplot(122)

    #save the number of doublets
    start = datetime.now()

    doublets_binned_sky = get_doublet_list(files_binned_sky)
    doublets_binned_targets = get_doublet_list(files_binned_targets)

    print(doublets_binned_sky)
    print(doublets_binned_targets)

    print('All doublets counted in', datetime.now() - start, 's')

    #histograms the number of doublets
    binned_sky_bins = np.append(np.arange(250, 425, 5), 425)
    binned_targets_bins = np.append(np.arange(2250, 2600, 10), 2600)

    doublets_bs_bin_centers, doublets_bs_bin_content, doublets_bs_bin_error = data_2_binned_errorbar(doublets_binned_sky, binned_sky_bins, binned_sky_bins[0], binned_sky_bins[-1], np.ones(len(doublets_binned_sky)), False)
    doublets_bt_bin_centers, doublets_bt_bin_content, doublets_bt_bin_error = data_2_binned_errorbar(doublets_binned_targets, binned_targets_bins, binned_targets_bins[0], binned_targets_bins[-1], np.ones(len(doublets_binned_sky)), False)

    #plot the doublets
    ax_doublet_binned_sky_dist.errorbar(doublets_bs_bin_centers, doublets_bs_bin_content, yerr = doublets_bs_bin_error, mfc = 'white', mec = 'tab:blue', marker = 'o', markersize = 3, linestyle = 'None', linewidth = 1, label = r'Iso. samples: $\mu = %.2f$, $\sigma = %.2f$' % (doublets_binned_sky.mean(), doublets_binned_sky.std()))
    ax_doublet_binned_sky_dist.hist(doublets_binned_sky, bins = binned_sky_bins, color = 'tab:blue', histtype = 'step')

    ax_doublet_binned_targets_dist.errorbar(doublets_bt_bin_centers, doublets_bt_bin_content, yerr = doublets_bt_bin_error, color = 'tab:blue', marker = 'o', markersize = 3, linestyle = 'None', linewidth = 1, label = 'Iso. samples')
    ax_doublet_binned_targets_dist.hist(doublets_binned_targets, bins = binned_targets_bins, color = 'tab:blue', histtype = 'step')

    #define the style of the axis
    ax_doublet_binned_sky_dist = set_style(ax_doublet_binned_sky_dist, '', r'$N_{\tau}$', r'Number of samples', 14)
    ax_doublet_binned_targets_dist = set_style(ax_doublet_binned_targets_dist, '', r'$N_{\tau}$', r'Number of samples', 14)

    #plot legends
    ax_doublet_binned_sky_dist.legend(loc = 'upper left', fontsize = 14)
    ax_doublet_binned_targets_dist.legend(loc = 'upper left', fontsize = 14)

    fig_doublet_dist.tight_layout()
    fig_doublet_dist.savefig(os.path.join(output_path, 'Doublets_distribution.pdf'), dpi=1000)
