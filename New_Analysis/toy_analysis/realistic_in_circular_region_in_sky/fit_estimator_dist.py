import pandas as pd
import numpy as np
import healpy as hp
from healpy.newvisufunc import projview

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

import scipy.interpolate as spline

sys.path.append('../src/')

import hist_manip
import fit_routines

from hist_manip import data_2_binned_errorbar
from hist_manip import get_bin_centers

from event_manip import compute_directional_exposure
from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import get_normalized_exposure_map
from event_manip import get_skymap

from axis_style import set_style

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#rounds the declination to multiples of 2 (the bin width)
def round_declination(dec):

    return dec - (dec % 2)

#rounds the rate to multiples of .5
def round_rate(rate):

    return rate - (rate % .5)

#fit the tail of the distribution of lambda
def get_fit_params_lambda_dist_per_rate(filename):

    #get the lambda distribution
    lambda_data = pd.read_json(filename)

    lambda_data = lambda_data[lambda_data['rate_upper_edges'] > 1]

    #make sure lists are arrays
    lambda_data['lambda_bin_centers'] = lambda_data['lambda_bin_centers'].apply(lambda x: np.array(x))
    lambda_data['lambda_bin_content'] = lambda_data['lambda_bin_content'].apply(lambda x: np.array(x))

    #compute declination bin centers
    rate_bin_low_edges = lambda_data['rate_low_edges'].to_numpy()
    rate_bin_upper_edges = lambda_data['rate_upper_edges'].to_numpy()

    #compute the error of each bin content
    lambda_data['lambda_bin_error'] = lambda_data['lambda_bin_content'].apply(lambda x: np.sqrt(x))

    #fit the lambda distribution as a function of declination
    lambda_data['tail_fit_params'] = lambda_data.apply(lambda x: fit_routines.perform_fit_exp(x['lambda_bin_centers'], x['lambda_bin_content'], x['lambda_bin_error'], x['lambda_dist_quantile_99']), axis=1)

    #save the comulative and normalized lambda CDF, along with fit initial point and beta_Lambda
    lambda_data['lambda_tail_fit_initial_point'] = lambda_data['tail_fit_params'].apply(lambda x: x[2][0])
    lambda_data['lambda_tail_fit_slope'] = lambda_data['tail_fit_params'].apply(lambda x: [x[0][1], x[1][1]])
    lambda_data['lambda_tail_fit_norm'] = lambda_data['tail_fit_params'].apply(lambda x: [x[0][0], x[1][0]])
    lambda_data['lambda_tail_fit_chi2'] = lambda_data['tail_fit_params'].apply(lambda x: x[4])

    #delete unnecessary columns
    lambda_data = lambda_data.drop('tail_fit_params', axis = 1)

    #update bin content
    lambda_data['above_fit_initial_point'] = lambda_data.apply(lambda x: x['lambda_bin_centers'] > x['lambda_tail_fit_initial_point'], axis = 1)
    lambda_data['updated_lambda_bin_content'] = lambda_data.apply(lambda x: np.where(x['above_fit_initial_point'], x['lambda_tail_fit_norm'][0]*np.exp(- x['lambda_tail_fit_slope'][0]*x['lambda_bin_centers']), x['lambda_bin_content']), axis = 1)
    lambda_data['cdf_lambda_bin_content'] = lambda_data['updated_lambda_bin_content'].apply(lambda x: np.cumsum(x) / np.sum(x))

    lambda_data = lambda_data.drop('above_fit_initial_point', axis = 1)

    #save distributions and fit parameters
    lambda_bin_centers = lambda_data['lambda_bin_centers'].to_numpy()
    lambda_bin_content = lambda_data['lambda_bin_content'].to_numpy()
    lambda_bin_error = lambda_data['lambda_bin_error'].to_numpy()
    lambda_cdf_bin_content = lambda_data['cdf_lambda_bin_content'].to_numpy()

    lambda_tail_fit_initial_point = lambda_data['lambda_tail_fit_initial_point'].to_numpy()
    lambda_tail_fit_slope = lambda_data['lambda_tail_fit_slope'].to_numpy()
    lambda_tail_fit_norm = lambda_data['lambda_tail_fit_norm'].to_numpy()
    #lambda_tail_fit_shift = lambda_data['lambda_tail_fit_shift'].to_numpy()
    lambda_tail_fit_chi2 = lambda_data['lambda_tail_fit_chi2'].to_numpy()

    #save fitted Lambda distribution in original file
    lambda_data.to_json(filename)

    return rate_bin_low_edges, rate_bin_upper_edges, [lambda_bin_centers, lambda_bin_content, lambda_bin_error, lambda_cdf_bin_content], [lambda_tail_fit_initial_point, np.array(lambda_tail_fit_norm, dtype=object), np.array(lambda_tail_fit_slope, dtype=object), lambda_tail_fit_chi2]

#get the axis with the declination values corresponding to ticks of declination
def get_exposure_dec_axis(ax_dec, ax_exposure, nticks, theta_max, pao_lat):

    #define the axis with nticks equally spaced values of declination
    dec_max = np.degrees(theta_max + pao_lat)

    dec_max = dec_max + (dec_max % 5)

    dec_ticks = np.linspace(-90, dec_max, nticks)

    #compute the corresponding directional exposure
    dir_exposure_ticks = compute_directional_exposure(np.radians(dec_ticks), theta_max, pao_lat)

    #normalize exposure
    dec_array = np.radians(np.linspace(-90, 90, 1000))
    dir_exposure = compute_directional_exposure(dec_array, theta_max, pao_lat)
    integrated_exposure = np.trapz(dir_exposure*np.cos(dec_array), x=dec_array)

    dir_exposure_ticks = dir_exposure_ticks / integrated_exposure

    #set ticks and tick labels
    ax_dec.set_xticks(dec_ticks)
    ax_dec.set_xticklabels(['%.0f' % dec for dec in dec_ticks])
    #ax_dec.margins(x=.05, y=.05)
    ax_exposure.set_xticks(dir_exposure_ticks)
    ax_exposure.set_xticklabels(['%.1f' % dir_exposure for dir_exposure in dir_exposure_ticks])
    ax_exposure.invert_xaxis()
    #ax_exposure.margins(x=0, y=0)

    return ax_dec, ax_exposure

#define the main function
if __name__ == '__main__':

    #save name of output path and creates it is does not exist
    output_path='./results/' + os.path.splitext(os.path.basename(sys.argv[0]))[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #save file containing distribution of lambda as a function of rate
    input_path = './datasets/lambda_dist'

    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    file_lambda_dist = 'Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_corrected_lambda_dist = 'Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_lambda_dist = os.path.join(input_path, file_lambda_dist)
    file_corrected_lambda_dist = os.path.join(input_path, file_corrected_lambda_dist)

    #check if both requested file exist
    if (not os.path.exists(file_lambda_dist)) or (not os.path.exists(file_lambda_dist)):
        print('One of the requested files does not exist!')
        exit()

    #save the corresponding dataframes
    lambda_dist_per_mu = pd.read_json(file_lambda_dist)
    corrected_lambda_dist_per_mu = pd.read_json(file_corrected_lambda_dist)

    print(lambda_dist_per_mu)

    #set position of the pierre auger observatory
    lat_pao = np.radians(-35.15) # this is the average latitude
    long_pao = np.radians(-69.2) # this is the averaga longitude
    height_pao = 1425*u.meter # this is the average altitude

    #define the earth location corresponding to pierre auger observatory
    pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

    #define theta_max
    theta_max = np.radians(80)

    # ------------------------------------------
    # Plot the distribution of Lambda and corrected Lambda for each mu value
    # ------------------------------------------
    #initialize figure
    fig_lambda_dist = plt.figure(figsize = (15, 4))

    ax_lambda_dist = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=2, fig = fig_lambda_dist)
    ax_corrected_lambda_dist = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=2, fig = fig_lambda_dist)
    ax_beta_mu_func = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_mean_mu_func = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)

    #get the colormap
    colormap = plt.get_cmap('RdBu')

    #produce the color array
    color_array = (lambda_dist_per_mu['mu_low_edges'].values - lambda_dist_per_mu['mu_low_edges'].min()) / (lambda_dist_per_mu['mu_low_edges'].max() - lambda_dist_per_mu['mu_low_edges'].min())
    color_array = colormap(color_array)

    #plot the distribution of lambda per expected value of events
    for i in range(len(lambda_dist_per_mu)):

        #save correspoding value of mu
        mu_low_edge = lambda_dist_per_mu['mu_low_edges'].loc[i]

        #save the bins edges of the lambda distribution
        lambda_bin_edges = np.array(lambda_dist_per_mu['lambda_bin_edges'].loc[i])
        lambda_bin_content = np.array(lambda_dist_per_mu['lambda_bin_content'].loc[i])

        #compute bin centers
        lambda_bin_centers = get_bin_centers(lambda_bin_edges)

        #compute bin errors
        lambda_bin_error = np.sqrt(lambda_bin_content)

        #normalize distribution
        integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

        lambda_bin_content = lambda_bin_content / integral
        lambda_bin_error = lambda_bin_error / integral

        #plot distribution for some cases
        #if mu_low_edge in [20, 22, 24]:
        ax_lambda_dist.plot(lambda_bin_centers, lambda_bin_content, color = color_array[i], marker = 'o', markersize = 5, mfc = 'white')

    #plot the distribution of lambda per expected value of events
    for i in range(len(corrected_lambda_dist_per_mu)):

        #save correspoding value of mu
        mu_low_edge = corrected_lambda_dist_per_mu['mu_low_edges'].loc[i]

        #save the bins edges of the lambda distribution
        lambda_bin_edges = np.array(corrected_lambda_dist_per_mu['lambda_bin_edges'].loc[i])
        lambda_bin_content = np.array(corrected_lambda_dist_per_mu['lambda_bin_content'].loc[i])

        #compute bin centers
        lambda_bin_centers = get_bin_centers(lambda_bin_edges)

        #compute bin errors
        lambda_bin_error = np.sqrt(lambda_bin_content)

        #normalize distribution
        integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

        lambda_bin_content = lambda_bin_content / integral
        lambda_bin_error = lambda_bin_error / integral

        #plot distribution for some cases
        #if mu_low_edge in [20, 22, 24]:
        ax_corrected_lambda_dist.plot(lambda_bin_centers, lambda_bin_content, color = color_array[i], marker = 'o', markersize = 5)

    #define the style of the plot
    ax_lambda_dist.set_yscale('log')
    ax_corrected_lambda_dist.set_yscale('log')

    ax_lambda_dist = set_style(ax_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)
    ax_corrected_lambda_dist = set_style(ax_corrected_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)

    fig_lambda_dist.tight_layout()
    fig_lambda_dist.savefig(os.path.join(output_path, 'Lambda_distribution_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))

    #save number of events
    # n_events = 1e5 #make this more flexible in the future
    # obs_time = 1 #in decades
    # rate = n_events / obs_time #in events per decade
    # target_radius = np.radians(1)
    # area_of_target = 2*np.pi*(1 - np.cos(target_radius))

    # fig_lambda_dist = plt.figure(figsize=(10, 4))
    # ax_lambda_dist = fig_lambda_dist.add_subplot(121)
    # ax_lambda_tail_slope = fig_lambda_dist.add_subplot(122)

    # #save the color map
    # colormap = plt.get_cmap('RdBu') #.reversed()
    #
    # #color_array = np.linspace(0, 1, len(rate_bin_low_edges))
    #
    # for i, rate_low_edge in enumerate(rate_bin_low_edges):
    #
    #     #save rate upper edge
    #     rate_upper_edge = rate_bin_upper_edges[i]
    #
    #     #define the fit range and save fit parameters
    #     fit_range = np.linspace(lambda_fit[0][i], max(lambda_dist[0][i]), 1000)
    #     fit_norm = lambda_fit[1][i][0]
    #     fit_slope = lambda_fit[2][i][0]
    #
    #     #plot the lambda distribution
    #     if rate_upper_edge % 3 == 0:
    #
    #         ax_lambda_dist.errorbar(lambda_dist[0][i], lambda_dist[1][i] / sum(lambda_dist[1][i]), yerr=lambda_dist[2][i] / sum(lambda_dist[1][i]), color = color_map(color_array[i]), alpha = .5, linewidth = 1, linestyle='None', marker='o', markersize=1)
    #         ax_lambda_dist.plot(fit_range, (fit_norm / sum(lambda_dist[1][i]))*np.exp(-fit_slope*fit_range), color = color_map(color_array[i]))
    #
    #
    #         #style of axis to plot lambda distribution
    #         ax_lambda_dist = set_style(ax_lambda_dist, '', r'$\Lambda$', 'Prob. density', 12)
    #         ax_lambda_dist.set_yscale('log')
    #         ax_lambda_dist.set_ylim(1e-7,.5)
    #
    #         cb_lambda_dist = fig_lambda_dist.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=min(rate_bin_low_edges), vmax=max(rate_bin_upper_edges)), cmap=color_map), ax=ax_lambda_dist) #, cmap=color_map)
    #         cb_lambda_dist.ax.set_ylabel(r'$\Gamma \;(\mathrm{decade}^{-1})$', fontsize=12)
    #
    #         not_min_slope = lambda_fit_tail_slope != min(lambda_fit_tail_slope)
    #
    #         #plot slope as a function of caracteristic time
    #         ax_lambda_tail_slope.errorbar(rate_bin_centers[not_min_slope], lambda_fit_tail_slope[not_min_slope], yerr=lambda_fit_tail_slope_error[not_min_slope], linestyle='None', marker='o', markersize=3) #, label = r'$\delta \in [%.0f^\circ, %.0f^\circ]$' % (omega_low, omega_high))
    #         ax_lambda_tail_slope = set_style(ax_lambda_tail_slope, '', r'$\Gamma \;(\mathrm{decade}^{-1})$', r'$\beta$', 12)
    #
    #         fig_lambda_dist.tight_layout()
    #         fig_lambda_dist.savefig(os.path.join(output_path, 'Lambda_distribution_IsotropicSkies_nEvent_%i_targetRadius_%.2f_th%.0f.pdf' % (n_events, np.degrees(target_radius), np.degrees(theta_max)) ))
    #
    #
    #         #define NSIDE parameter
    #         NSIDE = 128
    #
    #         #compute exposure map
    #         rate_map = rate*area_of_target*get_normalized_exposure_map(NSIDE, theta_max, lat_pao)
    #
    #         null_rate = (rate_map == 0)
    #         rate_map[null_rate] = hp.UNSEEN
    #         rate_map = hp.ma(rate_map)
    #
    #         #save figure
    #         fig_rate_skymap = plt.figure(figsize=(10,8)) #create figure
    #
    #         #plot sky map
    #         projview(
    #         rate_map,
    #         override_plot_properties={'figure_size_ratio' : .6},
    #         graticule=True,
    #         graticule_labels=True,
    #         #title=r"$\omega(\alpha, \delta)$ for $\theta_{\max} = %.0f^\circ$" % np.degrees(theta_max),   #unit=r"$\log_{10}$ (Number of events)",
    #         xlabel=r"$\alpha$",
    #         ylabel=r"$\delta$",
    #         cmap='coolwarm',
    #         cb_orientation="horizontal",
    #         projection_type="hammer",
    #         fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
    #         longitude_grid_spacing = 30,
    #         latitude_grid_spacing = 15,
    #         xtick_label_color='black',
    #         min=0,
    #         max=18.5,
    #         unit = r'$\Gamma_{\mathrm{target}}(\delta) \;({\mathrm{decade}^{-1}})$',
    #         )
    #
    #         plt.savefig(os.path.join(output_path, 'Skymap_event_rate_nEvents_%i_targetRadius_%.2f_FullEfficiency_th%.0f.pdf' % (n_events, np.degrees(target_radius), np.degrees(theta_max))), dpi=1000)
