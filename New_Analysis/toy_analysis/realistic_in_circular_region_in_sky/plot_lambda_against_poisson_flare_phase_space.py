import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.interpolate import Akima1DInterpolator as akima_spline

#for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from astropy.time import Time

import sys
import os
import pickle

sys.path.append('../src/')

from hist_manip import data_2_binned_errorbar
#from fit_routines import perform_fit_exp

from axis_style import set_style
from axis_style import set_cb_style

#enable latex rendering and latex like style font
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#compute n_samples of n_events distributed uniformly in time
# def get_time_stamps(n_samples, n_events, obs_time):
#
#     #compute matrix of events times
#     event_time = np.random.randint(0, obs_time, size = (n_samples, n_events))
#
#     return event_time
#
# #compute times of events from flare
# def get_flare_time_stamps(n_samples, n_events, flare_start, flare_duration):
#
#     #compute matrix of events times
#     event_time = np.random.randint(flare_start, flare_start + flare_duration, size = (n_samples, n_events))
#
#     return event_time
#
# #compute array given a matrix of time stamps
# def compute_lambda(event_time, exp_rate):
#
#     #compute lambda
#     event_time = np.sort(event_time, axis = 1)
#     time_diff = np.diff(event_time, axis = 1)*exp_rate
#
#     lambda_array = -np.sum(np.log(time_diff), axis = 1)
#
#     return lambda_array
#
# #computes p_values for the lambda_array
# def get_lambda_pvalues(lambda_array, lambda_dist, fit_initial, tail_slope, fit_scale):
#
#     #initialize p_value array
#     lambda_pvalues = np.zeros(len(lambda_array))
#
#     #saves the lambda_bin_centers and bin_contents
#     lambda_bin_centers = np.array(lambda_dist[0])
#     lambda_bin_content = np.array(lambda_dist[1])
#
#     #compute the discrete cdf of lambda and interpolate it
#     below_fit_init = lambda_bin_centers <= fit_initial
#     lambda_bins = lambda_bin_centers[below_fit_init]
#     discrete_cdf_lambda = np.cumsum(lambda_bin_content[below_fit_init]) / np.sum(lambda_bin_content)
#
#     interpolated_cdf_lambda = akima_spline(lambda_bins, discrete_cdf_lambda)
#
#     #if lambda_value is below initial point the p_value used the interpolated discrete cdf
#     lambda_below_fit_init = lambda_array < fit_initial
#     lambda_pvalues[lambda_below_fit_init] = 1 - interpolated_cdf_lambda(lambda_array[lambda_below_fit_init])
#
#     #if lambda_value is above initial point the p_value is analytical
#     lambda_above_fit_init = lambda_array >= fit_initial
#
#     #print(np.exp(-tail_slope*lambda_array[lambda_above_fit_init]))
#     #const = (fit_scale / tail_slope)*np.exp(-tail_slope*fit_initial)
#     lambda_pvalues[lambda_above_fit_init] = (1 - discrete_cdf_lambda[-1])*np.exp(-tail_slope*(lambda_array[lambda_above_fit_init] - fit_initial)) # - fit_initial))
#
#     #print('Discrete p_value:', np.log10(1 - discrete_cdf_lambda[-1]))
#     #print('Interpolated p_value:', np.log10(1 - interpolated_cdf_lambda(fit_initial)))
#     #print('Fitted p_value', np.log10(np.exp(-tail_slope*fit_initial)) )
#
#     return lambda_pvalues
#
# #function to print the content of each bin in 2d histogram
# def print_heatmap_bin_content(ax, x_bins, y_bins, bin_content, fontsize):
#
#     for i, x_pos in enumerate(x_bins):
#
#         for j, y_pos in enumerate(y_bins):
#
#             ax.text(x_pos, y_pos, r'$%.1f$' % bin_content[i, j], ha="center", va="center", fontsize = fontsize)
#
#     return ax

#define a function to get a list of files from a path and a pattern
def get_filelist(input_path, pattern):

    filelist = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and pattern in filename:

            filelist.append(filename)

    return filelist

#function to fetch to sum the lattices of either estimators or pvalues
def merge_samples(filelist):

    #initialize list
    all_samples = []

    for file in filelist:

        #save the array with values to be summed
        with open(file, 'rb') as f:
            value_grid_per_sample = pickle.load(f)

        all_samples.append(value_grid_per_sample)

    result = np.concatenate(all_samples, axis = 3)

    return result

#function to count the fraction of samples for each the pvalue computed with lambda is smaller than the one computed with poisson
def get_lambda_performance(pvalues_flares):

    #compute the difference between pvalues with poisson and lambda
    pvalue_diff = pvalues_flares[:,:,1,:] - pvalues_flares[:,:,0,:]
    pvalue_diff_corr_lambda = pvalues_flares[:,:,2,:] - pvalues_flares[:,:,0,:]

    #counts the number of elements that are negative
    negative_values = np.sum(pvalue_diff < 0, axis = 2)
    negative_values_corr_lambda = np.sum(pvalue_diff_corr_lambda < 0, axis = 2)

    #transform these values into a fraction
    frac_lambda = negative_values / pvalues_flares.shape[3]
    frac_lambda_corr = negative_values_corr_lambda / pvalues_flares.shape[3]

    #transform this fraction into a more meaningful number
    performance_factor_lambda = 100*frac_lambda #np.log10(1 - frac_lambda)
    performance_factor_lambda_corr = 100*frac_lambda_corr #np.log10(1 - frac_lambda_corr)

    return performance_factor_lambda, performance_factor_lambda_corr

if __name__ == '__main__':

    #define a few constants
    time_begin = Time('2010-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    time_end = Time('2020-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    obs_time = time_end - time_begin

    #define the input directory
    input_path = './datasets/flare_lattice_study'

    #define the output directory
    output_path = './results/lambda_performance'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #get lists of files with iso and (iso + flare) samples
    filelist_flare_info = 'Info'
    filelist_iso_estimator = 'Estimators_IsoDist'
    filelist_iso_pvalues = 'PValues_IsoDist'
    filelist_flare_estimator = 'Estimators_FlareLattice'
    filelist_flare_pvalues = 'PValues_FlareLattice'

    filelist_flare_info = get_filelist(input_path, filelist_flare_info)
    filelist_iso_estimator = get_filelist(input_path, filelist_iso_estimator)
    filelist_iso_pvalues = get_filelist(input_path, filelist_iso_pvalues)
    filelist_flare_estimator = get_filelist(input_path, filelist_flare_estimator)
    filelist_flare_pvalues = get_filelist(input_path, filelist_flare_pvalues)

    if np.any([len(filelist_flare_info), len(filelist_iso_estimator), len(filelist_iso_pvalues), len(filelist_flare_estimator), len(filelist_flare_pvalues)] == 0):
        print('At least one of the requested lists of files is empty')
        exit()

    #get the grid of flare intensities and durations
    with open(filelist_flare_info[0], 'rb') as file:
        flare_intensity, flare_duration = pickle.load(file)

    #transform flare duration into a more meaningful quantity
    flare_duration = np.log10(flare_duration / obs_time)

    #get the merged distributions of pvalues and estimators
    estimators_flares = merge_samples(filelist_flare_estimator)
    pvalues_flares = merge_samples(filelist_flare_pvalues)

    #get the number of samples for each pvalue with lambda is smaller than with poisson
    performance_factor_lambda, performance_factor_lambda_corr = get_lambda_performance(pvalues_flares)

    #save colormap
    colormap = plt.get_cmap('RdBu_r')

    #----------------------------
    # Plotting
    #----------------------------
    #initialize figure and axis
    fig_lambda_performance_against_poisson = plt.figure(figsize=(10, 4))

    ax_lambda_performance = fig_lambda_performance_against_poisson.add_subplot(1, 2, 1)
    ax_corrected_lambda_performance = fig_lambda_performance_against_poisson.add_subplot(1, 2, 2)

    #ax_lambda_performance = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2, fig=fig_lambda_performance_against_poisson)
    #ax_lambda_dist_1day = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1, fig=fig_lambda_performance)
    #ax_lambda_dist_1month = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1, fig=fig_lambda_performance)
    #ax_lambda_pvalue_dist_1day = plt.subplot2grid((2, 4), (1, 2), colspan=1, rowspan=1, fig=fig_lambda_performance)
    #ax_lambda_pvalue_dist_1month = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1, fig=fig_lambda_performance)

    #compute lambda distributions for different flare intensities and times
    duration_1day = 86_164
    duration_1month = 30*86_164

    # lambda_bins = 100
    # lambda_pv_bins = 50
    # lambda_lower = lambda_dist_bin_centers[0]
    # lambda_upper = lambda_dist_bin_centers[-1]
    #to plot countour curves to assess lambda performance against the simple number of events

    #events_per_flare_grid, flare_duration_grid = np.meshgrid(events_per_flare_array, np.log10(flare_duration_array / obs_time))

    #print(events_per_flare_grid[0,:])
    #print(flare_duration_grid[:,0])
    #lower_contours = np.arange(-4.25, np.log10(1 - poisson_pvalue), .25)
    #upper_contours = np.linspace(np.log10(1 - poisson_pvalue), 0, 5) #, 0)
    #contour_levels = np.concatenate((lower_contours, upper_contours))

    #compute the limits of the lambda performance
    lambda_performance_lower = 0 #-np.log10(pvalues_flares.shape[3])
    lambda_performance_upper = 100
    lambda_performance_contour_levels = np.append(np.arange(lambda_performance_lower, lambda_performance_upper, 5), lambda_performance_upper)

    contour_lambda_performance = ax_lambda_performance.contourf(flare_duration, flare_intensity, np.transpose(performance_factor_lambda), levels = lambda_performance_contour_levels, cmap = colormap) #norm = mcolors.TwoSlopeNorm(vmin = contour_levels[0], vcenter = np.log10(1 - poisson_pvalue), vmax =contour_levels[-1]))
    contour_corrected_lambda_performance = ax_corrected_lambda_performance.contourf(flare_duration, flare_intensity, np.transpose(performance_factor_lambda_corr), levels = lambda_performance_contour_levels, cmap = colormap)

    #plot the contour corresponding to 0
    poisson_contour_lambda_performance = ax_lambda_performance.contour(flare_duration, flare_intensity, np.transpose(performance_factor_lambda), levels = [50], colors = 'black', linestyles = 'dashed', linewidths = 1)

    #define the style of the axis
    #ax_lambda_performance = set_style(ax_lambda_performance, r'$\mu = %.0f$ events, $N_{\mathrm{events}} = %.0f$, $T_{\mathrm{obs}} = 10$ years' % (exp_nEvents, iso_nEvents), r'$\log_{10} \left( \Delta t_{\mathrm{flare}} / 10 \mathrm{\,years} \right)$', r'$n_{\mathrm{events}}$', 12)
    #ax_lambda_performance = print_heatmap_bin_content(ax_lambda_performance, flare_duration_grid[:,0], events_per_flare_grid[0,:], np.transpose(frac_lambdaPvalues_below_poissonPvalue), 10)

    #create and define style of color bar
    cb_lambda_performance = fig_lambda_performance_against_poisson.colorbar(contour_lambda_performance, ax=ax_lambda_performance)
    cb_lambda_performance = set_cb_style(cb_lambda_performance, r'$\log_{10} \left( 1 - \displaystyle \frac{n_{\Lambda}}{n_{\mathrm{samples}}} \right)$', [lambda_performance_lower, lambda_performance_upper], 12)

    #labeled_contour_levels = np.linspace(-4, 0, 5)
    #cb_lambda_performance.set_ticks(labeled_contour_levels)
    #cb_lambda_performance.set_ticklabels(['%.1f' % label for label in labeled_contour_levels], fontsize = 12)

    #cb_lambda_performance.ax.hlines(np.log10(1 - poisson_pvalue), 0, 1, color = 'black', linestyle = 'solid')
    #ax_lambda_performance.clabel(poisson_contour_lambda_performance, fontsize = 12, fmt='%.0f \%%')

    #plot axis with intuitive time scales
    intuitive_duration_array = np.log10(np.array([1, 7, 30, 366]) * 86_164 / obs_time) # in seconds
    intuitive_duration_array_label = ['1 day', '1 week', '1 month', '1 year']

    ax_intuitive_duration = ax_lambda_performance.twiny()

    ax_intuitive_duration.set_xlim(ax_lambda_performance.get_xlim())
    ax_intuitive_duration.set_xticks(intuitive_duration_array)
    ax_intuitive_duration.set_xticklabels(intuitive_duration_array_label)

    #plot vertical lines with the intuitive time scales
    ax_lambda_performance.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .5, linestyle = 'dashed')

    #plot legend
    #legend_handles = [Line2D([0], [0], color=color_list[j], linewidth=1, label=f'$n = {flare_intensity}$') for j, flare_intensity in enumerate(events_per_flare_array[::2])]
    #legend_labels = [r'$n = %i$' % flare_intensity for flare_intensity in events_per_flare_array[::2]]

    #fig_lambda_performance.legend(handles = legend_handles, labels = legend_labels, fontsize = 12, loc = 'upper center', ncols = 5, columnspacing = 1., handlelength = 1., handletextpad = .5, bbox_to_anchor=(0.75, 1), frameon = False)

    fig_lambda_performance_against_poisson.tight_layout()
    fig_lambda_performance_against_poisson.savefig(os.path.join(output_path, 'LambdaPValues_below_poissonPvalue.pdf'))

    # color_list = color_map(np.linspace(0.2, .9, len(events_per_flare_array[::2])))
    #
    # #find the array of lambda and lambda_pvalues for each intensity and duration
    # lambda_dist_1day = all_lambda_tensor[j, np.searchsorted(flare_duration_array, duration_1day),:]
    # lambda_dist_1month = all_lambda_tensor[j, np.searchsorted(flare_duration_array, duration_1month),:]
    #
    # lambda_pvalue_dist_1day = all_lambda_pvalues_tensor[j, np.searchsorted(flare_duration_array, duration_1day),:]
    # lambda_pvalue_dist_1month = all_lambda_pvalues_tensor[j, np.searchsorted(flare_duration_array, duration_1month),:]
    #
    # #build distributions
    # lambda_1day_bin_centers, lambda_1day_bin_content, lambda_1day_bin_error = data_2_binned_errorbar(lambda_dist_1day, lambda_bins, lambda_lower, lambda_upper, np.ones(n_samples), False)
    # lambda_1month_bin_centers, lambda_1month_bin_content, lambda_1month_bin_error = data_2_binned_errorbar(lambda_dist_1month, lambda_bins, lambda_lower, lambda_upper, np.ones(n_samples), False)
    #
    # lambda_pv_1day_bin_centers, lambda_pv_1day_bin_content, lambda_pv_1day_bin_error = data_2_binned_errorbar(np.log10(lambda_pvalue_dist_1day), lambda_pv_bins, -10, 0, np.ones(n_samples), False)
    # lambda_pv_1month_bin_centers, lambda_pv_1month_bin_content, lambda_pv_1month_bin_error = data_2_binned_errorbar(np.log10(lambda_pvalue_dist_1month), lambda_pv_bins, -10, 0, np.ones(n_samples), False)
    #
    # #plot distributions
    # ax_lambda_dist_1day.plot(lambda_1day_bin_centers, lambda_1day_bin_content / np.trapz(lambda_1day_bin_content, x = lambda_1day_bin_centers), color = color_list[j], linewidth = 1)
    # ax_lambda_dist_1month.plot(lambda_1month_bin_centers, lambda_1month_bin_content / np.trapz(lambda_1month_bin_content, x = lambda_1month_bin_centers), color = color_list[j], linewidth = 1)
    #
    # ax_lambda_pvalue_dist_1day.plot(lambda_pv_1day_bin_centers, lambda_pv_1day_bin_content / np.trapz(lambda_pv_1day_bin_content, x = lambda_pv_1day_bin_centers), color = color_list[j], linewidth = 1)
    # ax_lambda_pvalue_dist_1month.plot(lambda_pv_1month_bin_centers, lambda_pv_1month_bin_content / np.trapz(lambda_pv_1month_bin_content, x = lambda_pv_1month_bin_centers), color = color_list[j], linewidth = 1)
    #
    #
    # #plot nominal distributions
    # ax_lambda_dist_1day.plot(lambda_dist_bin_centers[::10], lambda_dist_bin_content[::10] / lambda_dist_integral, color = 'gray', linestyle = 'dashed', linewidth = 1)
    # ax_lambda_dist_1month.plot(lambda_dist_bin_centers[::10], lambda_dist_bin_content[::10] / lambda_dist_integral, color = 'gray', linestyle = 'dashed', linewidth = 1)
    #
    # pvalue_cont = np.linspace(-10, 0, 100)
    # pvalue_pdf_cont = np.log(10)*np.power(10, pvalue_cont)
    # ax_lambda_pvalue_dist_1day.plot(pvalue_cont, pvalue_pdf_cont, color = 'tab:gray', linestyle = 'dashed', linewidth = 1)
    # ax_lambda_pvalue_dist_1month.plot(pvalue_cont, pvalue_pdf_cont, color = 'tab:gray', linestyle = 'dashed', linewidth = 1)
    #
    # ax_lambda_pvalue_dist_1day.vlines(np.log10(poisson_pvalue), 1e-5, 5, color = 'black', linestyle = 'dashed', linewidth = 1)
    # ax_lambda_pvalue_dist_1month.vlines(np.log10(poisson_pvalue), 1e-5, 5, color = 'black', linestyle = 'dashed', linewidth = 1)
    #
    # #define the style of the plots
    # ax_lambda_dist_1day.set_yscale('log')
    # ax_lambda_dist_1month.set_yscale('log')
    #
    # ax_lambda_pvalue_dist_1day.set_ylim(1e-4, 5)
    # ax_lambda_pvalue_dist_1month.set_ylim(1e-4, 5)
    # ax_lambda_pvalue_dist_1day.set_yscale('log')
    # ax_lambda_pvalue_dist_1month.set_yscale('log')
    #
    # ax_lambda_dist_1day = set_style(ax_lambda_dist_1day, r'$\Delta t_{\mathrm{flare}} = 1$ day', r'$\Lambda$', r'Prob. density', 12)
    # ax_lambda_dist_1month = set_style(ax_lambda_dist_1month, r'$\Delta t_{\mathrm{flare}} = 1$ month', r'$\Lambda$', r'Prob. density', 12)
    # ax_lambda_pvalue_dist_1day = set_style(ax_lambda_pvalue_dist_1day, r'', r'$\log_{10} p_{\Lambda}$', r'Prob. density', 12)
    # ax_lambda_pvalue_dist_1month = set_style(ax_lambda_pvalue_dist_1month, r'', r'$\log_{10} p_{\Lambda}$', r'Prob. density', 12)
