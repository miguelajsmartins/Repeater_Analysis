import pandas as pd
import numpy as np
import healpy as hp
import math
from healpy.newvisufunc import projview, newprojplot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

import scipy.interpolate as spline
from scipy.stats import poisson

sys.path.append('./src/')
sys.path.append('.')

import hist_manip
from hist_manip import data_2_binned_errorbar
import event_manip

from event_manip import compute_directional_exposure
from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import get_normalized_exposure_map
from event_manip import get_skymap
from event_manip import colat_to_dec
from event_manip import dec_to_colat

from axis_style import set_style

from array_manip import unsorted_search

from compute_lambda_pvalue_binned_sky import compute_p_value

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

#----------------------------------------------------
# Plots for the proceedings of ICRC 2023
#----------------------------------------------------
#complains if no filename is given
if len(sys.argv) < 2:
    print('No filename was given!')
    exit()

#load data with the skymap postrial p values
penalized_pvalue_file = sys.argv[1]

#construct name of file given the name of file read
output_path = os.path.dirname(penalized_pvalue_file)
basename = os.path.splitext(os.path.basename(penalized_pvalue_file))[0]
split_name = basename.split('_')
split_path = output_path.split('/')

output_name_skymap = 'Skymap_' + split_name[0] + '_' + split_name[1] + '_pvalues_' + split_name[3] + '_' + split_path[3] + '.png'
output_name_dist = 'Dist_' + split_name[0] + '_' + split_name[1] + '_pvalues_' + split_name[3] + '_' + split_path[3] + '.png'

#save the number of flares, events per flare and flare duration
n_flares = split_name[9]
n_events = split_name[11]
flare_duration = split_name[13]

#save datafile with the position of the sources and the penalized pvalues
penalized_pvalue_data = pd.read_csv(penalized_pvalue_file)

#save the right ascension, declination of flares and penalized p values
ra_target = np.radians(penalized_pvalue_data['ra_target'].to_numpy())
dec_target = np.radians(penalized_pvalue_data['dec_target'].to_numpy())
poisson_penalized_pvalue = penalized_pvalue_data['poisson_penalized_pvalue'].to_numpy()
lambda_penalized_pvalue = penalized_pvalue_data['lambda_penalized_pvalue'].to_numpy()
colat_target = dec_to_colat(dec_target)

#compute log10 of p-value
log_poisson_penalized_pvalue = np.log10(poisson_penalized_pvalue)
log_lambda_penalized_pvalue = np.log10(lambda_penalized_pvalue)

#get the colormap
color_map = cm.get_cmap('coolwarm').reversed()

#define the limits of the color scale for plotting
upper_limit_cb = 0
lower_limit_cb = min([log_lambda_penalized_pvalue.min(), log_poisson_penalized_pvalue.min()])
lower_limit_cb = round_up(lower_limit_cb)
log_pvalue_range = upper_limit_cb - lower_limit_cb

#define the color of the markers for each pvalue, according to their value
color_log_poisson_pvalue = color_map(1+log_poisson_penalized_pvalue / log_pvalue_range)
color_log_lambda_pvalue = color_map(1+log_lambda_penalized_pvalue / log_pvalue_range)

# ----------
# plot skymaps of penalized pvalues
# ----------
#define figures to plot the p-values
fig_skymap_penalized_pvalues = plt.figure(figsize=(10, 4))
ax_skymap_poisson_pvalue = fig_skymap_penalized_pvalues.add_subplot(121, projection='hammer')
ax_skymap_lambda_pvalue = fig_skymap_penalized_pvalues.add_subplot(122, projection='hammer')

# plot skymaps of penalized pvalues
ax_skymap_poisson_pvalue.scatter(ra_to_phi(ra_target), dec_target, marker='o', color=color_log_poisson_pvalue)
ax_skymap_poisson_pvalue.grid()

ax_skymap_lambda_pvalue.scatter(ra_to_phi(ra_target), dec_target, marker='o', color=color_log_lambda_pvalue)
ax_skymap_lambda_pvalue.grid()

#define the style of axis
title = r'$n_{\mathrm{flares}} = %s$, $n_{\mathrm{events}} = %s$, $\Delta T_{\mathrm{flare}} = %s$ days' % (n_flares, n_events, flare_duration)
ax_skymap_lambda_pvalue = set_style(ax_skymap_lambda_pvalue, title, r'$\alpha$', r'$\delta$', 12)
ax_skymap_poisson_pvalue = set_style(ax_skymap_poisson_pvalue, title, r'$\alpha$', r'$\delta$', 12)

ax_skymap_poisson_pvalue.set_title(title, y=1.1)
ax_skymap_lambda_pvalue.set_title(title, y=1.1)

#plot color bar
cb_lambda_pvalue = fig_skymap_penalized_pvalues.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=lower_limit_cb, vmax=upper_limit_cb), cmap=color_map), ax= ax_skymap_lambda_pvalue, orientation='horizontal')
cb_lambda_pvalue.ax.set_xlabel(r'$\log_{10} (p_{\Lambda}^*\mathrm{-value})$', fontsize=12)

cb_poisson_pvalue = fig_skymap_penalized_pvalues.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=lower_limit_cb, vmax=upper_limit_cb), cmap=color_map), ax= ax_skymap_poisson_pvalue, orientation='horizontal')
cb_poisson_pvalue.ax.set_xlabel(r'$\log_{10} (p_{\mathrm{poisson}}^*\mathrm{-value})$', fontsize=12)

fig_skymap_penalized_pvalues.tight_layout()
fig_skymap_penalized_pvalues.savefig(os.path.join(output_path, output_name_skymap), dpi=1000)

# ----------
# plot distributon of penalized pvalues
# ----------
fig_pvalue_dist = plt.figure(figsize=(5, 4))
ax_pvalue_dist = fig_pvalue_dist.add_subplot(111)

#define the binning
poisson_pvalue_bin_centers, poisson_pvalue_bin_content, poisson_pvalue_bin_error = data_2_binned_errorbar(log_poisson_penalized_pvalue, 10, lower_limit_cb, upper_limit_cb, np.ones(len(ra_target)), False)
lambda_pvalue_bin_centers, lambda_pvalue_bin_content, lambda_pvalue_bin_error = data_2_binned_errorbar(log_lambda_penalized_pvalue, 10, lower_limit_cb, upper_limit_cb, np.ones(len(ra_target)), False)

#plot error bars
flare_color = { '1' : 'tab:red', '7' : 'tab:orange'}

ax_pvalue_dist.hist(log_poisson_penalized_pvalue, bins=10, range=[lower_limit_cb, upper_limit_cb], color = 'tab:blue', linestyle = 'dashed', histtype='step')
ax_pvalue_dist.hist(log_lambda_penalized_pvalue, bins=10, range=[lower_limit_cb, upper_limit_cb], color = flare_color[flare_duration], linestyle = 'solid', histtype='step')
ax_pvalue_dist.errorbar(poisson_pvalue_bin_centers, poisson_pvalue_bin_content, yerr=poisson_pvalue_bin_error, color='tab:blue', linestyle='none', marker='o', markersize=5, fillstyle='none', label='Poisson')
ax_pvalue_dist.errorbar(lambda_pvalue_bin_centers, lambda_pvalue_bin_content, yerr=lambda_pvalue_bin_error, color=flare_color[flare_duration], linestyle='none', marker='o', markersize=3, label=r'$\Lambda$')

ax_pvalue_dist = set_style(ax_pvalue_dist, '', r'$\log_{10} (p^*)$', r'Number of targets', 12)
ax_pvalue_dist.legend(loc='upper left', title=r'$n_{\mathrm{flares}} = %s$, $n_{\mathrm{events}} = %s$, \vspace{5pt}\par $\Delta T_{\mathrm{flare}} = %s$ days' % (n_flares, n_events, flare_duration), title_fontsize=12, fontsize=12)

fig_pvalue_dist.tight_layout()
fig_pvalue_dist.savefig(os.path.join(output_path, output_name_dist), dpi=1000)

# #get sample with flare events
# event_file = os.path.join(output_path, '_'.join(split_name[2:-3]) + '.parquet')
# event_data = pd.read_parquet(event_file, engine='fastparquet')
#
# #load file with pos-trial pvalues
# postrial_pvalue_data = pd.read_parquet(postrial_pvalue_file, engine='fastparquet')
#
# #define position of the observatory
# lat_pao = np.radians(-35.15) # this is the average latitude
# long_pao = np.radians(-69.2) # this is the averaga longitude
# height_pao = 1425*u.meter # this is the average altitude
#
# #define the earth location corresponding to pierre auger observatory
# pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)
#
# #define the maximum value of theta
# theta_max = np.radians(80)
#
# #compute exposure map to define excluded regions
# target_radius = np.radians(1)
# target_area = 2*np.pi*(1 - np.cos(target_radius))
#
# #load positions of sources
# total_events, flare_colat, flare_ra = get_sources_position(event_file)
# obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]
#
# rate = total_events / obs_time
#
# #compute penalized p-values for the targets
# lambda_dist_file = './datasets/estimator_dist/Lambda_dist_per_rate_997.json'
# lambda_dist_data = pd.read_json(lambda_dist_file)
#
# targeted_search_pvalue_data = compute_targeted_search_penalized_pvalues(event_data, colat_to_dec(flare_colat), flare_ra, lambda_dist_data, target_radius, rate, theta_max, lat_pao)
#
# #define the output name of p-values
# output_targeted_search = 'TargetedSearch_Penalized_' + '_'.join(split_name[2:]) + '.csv'
# output_targeted_search = os.path.join(output_path, output_targeted_search)
#
# #save output of targeted search as csv for incorporate in latex
# targeted_search_pvalue_data.to_csv(output_targeted_search, index = True)
#
# #save the target centers and postrial pvalues
# target_ra = np.radians(postrial_pvalue_data['ra_target'].to_numpy())
# target_colat = event_manip.dec_to_colat(np.radians(postrial_pvalue_data['dec_target'].to_numpy()))
# target_event_expectation = postrial_pvalue_data['expected_events_in_target'].to_numpy()
# target_poisson_postrial = postrial_pvalue_data['poisson_postrial_p_value'].to_numpy()
# target_lambda_postrial = postrial_pvalue_data['lambda_postrial_p_value'].to_numpy()
#
# #delete dataframe from memory
# del postrial_pvalue_data
#
# #read the value of the nside, number of flares, flare duration and events per flare parameter of used in the file
# NSIDE=int(split_name[-1])
# n_flares = int(split_name[9])
# n_events_per_flare = int(split_name[11])
# flare_duration = int(split_name[13])
#
# #produce skymap with pvalues
# npix = hp.nside2npix(NSIDE)
#
# skymap_postrial_poisson = np.zeros(npix)
# skymap_postrial_lambda = np.zeros(npix)
#
# pixel_indices = hp.ang2pix(NSIDE, target_colat, target_ra)
#
# #if postrial probabilties are null, then set them to their minimum number
# target_poisson_postrial[target_poisson_postrial == 0] = 1/991 #where 991 is the number of samples
# target_lambda_postrial[target_lambda_postrial == 0] = 1/991
#
# np.add.at(skymap_postrial_poisson, pixel_indices, np.log10(target_poisson_postrial))
# np.add.at(skymap_postrial_lambda, pixel_indices, np.log10(target_lambda_postrial))
#
# #save the color map
# color_map = cm.get_cmap('coolwarm').reversed()
#
#
#
# rate_map = total_events*target_area*get_normalized_exposure_map(NSIDE, theta_max, lat_pao)
#
# #exclude pixels outside FoV of observatory
# low_rate = (rate_map < 1)
#
# skymap_postrial_poisson[low_rate] = hp.UNSEEN
# skymap_postrial_poisson = hp.ma(skymap_postrial_poisson)
#
# skymap_postrial_lambda[low_rate] = hp.UNSEEN
# skymap_postrial_lambda = hp.ma(skymap_postrial_lambda)
#
# #save figure
# fig_skymap = plt.figure(figsize=(10,8)) #create figure
#
# #plot sky map for poisson
# hp.newvisufunc.projview(
#     skymap_postrial_poisson,
#     override_plot_properties={'figure_size_ratio' : .6},
#     graticule=True,
#     graticule_labels=True,
#     title=r"$n_{\mathrm{flares}} = %i$, $n_{\mathrm{events}} = %i$, $\Delta t_{\mathrm{flare}} = %i$ days" % (n_flares, n_events_per_flare, flare_duration),
#     xlabel=r"$\alpha$",
#     ylabel=r"$\delta$",
#     cmap=color_map,
#     cb_orientation="horizontal",
#     projection_type="hammer",
#     fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
#     longitude_grid_spacing = 30,
#     latitude_grid_spacing = 30,
#     xtick_label_color='black',
#     min=-3,
#     max=0,
#     unit = r'$\log_{10} (p_{\mathrm{poisson}}\mathrm{-value})$',
# );
#
# hp.newvisufunc.newprojplot(theta=flare_colat, phi=ra_to_phi(flare_ra), marker='o', linestyle = 'None', fillstyle='none', color = 'black', markersize=10)
#
# plt.savefig(os.path.join(output_path, output_name_poisson), dpi=1000)
#
# #plot skymap for lambda
# hp.newvisufunc.projview(
#     skymap_postrial_lambda,
#     override_plot_properties={'figure_size_ratio' : .6},
#     graticule=True,
#     graticule_labels=True,
#     title=r"$n_{\mathrm{flares}} = %i$, $n_{\mathrm{events}} = %i$, $\Delta t_{\mathrm{flare}} = %i$ days" % (n_flares, n_events_per_flare, flare_duration),
#     xlabel=r"$\alpha$",
#     ylabel=r"$\delta$",
#     cmap=color_map,
#     cb_orientation="horizontal",
#     projection_type="hammer",
#     fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
#     longitude_grid_spacing = 30,
#     latitude_grid_spacing = 30,
#     xtick_label_color='black',
#     min=-3,
#     max=0,
#     unit = r'$\log_{10} (p_{\Lambda} \mathrm{-value})$',
# );
#
# newprojplot(theta=flare_colat, phi=ra_to_phi(flare_ra), marker='o', linestyle = 'None', fillstyle='none', color = 'black', markersize=10)
#
# plt.savefig(os.path.join(output_path, output_name_lambda), dpi=1000)
