import numpy as np
import numpy.ma as ma
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

from event_manip import compute_directional_exposure

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

def get_mu_around_flare(ra_flare, dec_flare, target_radius, theta_max, pao_lat, n_events):

    #compute the expected number of events in the position of the flare
    dec_array = np.linspace(-np.pi / 2, np.pi / 2 , 1000)
    exposure_array = compute_directional_exposure(dec_array, theta_max, pao_lat)
    integrated_exposure = 2*np.pi*np.trapz(exposure_array*np.cos(dec_array), x = dec_array)

    target_area = 2*np.pi*(1 - np.cos(target_radius))
    exposure_at_flare = compute_directional_exposure([dec_flare], theta_max, pao_lat) / integrated_exposure

    mu_at_flare = n_events*exposure_at_flare*target_area
    mu_at_flare = mu_at_flare[0]

    return mu_at_flare

#define a function to get a list of files from a path and a pattern
def get_filelist(input_path, pattern):

    filelist = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and pattern in filename:

            filelist.append(filename)

    return np.array(filelist)

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

#get pos-trial p-values
def compute_postrial_pvalues(pvalues_flares, pvalues_iso, estimator_type):

    start = datetime.now()

    #set up dictionary to convert estimator name into an index
    estimator_dic = {'poisson' : 0, 'lambda' : 1, 'corrected_lambda' : 2}

    try:
        index = estimator_dic[estimator_type]
    except KeyError:
        print('Requested estimator name: (%s) does not exist in available keys' % estimator_type, list(estimator_dic.keys()))
        exit()

    #save the values of pvalues for easier manipulation
    pvalues_flare = pvalues_flares[:,:,index,:]
    pvalues_iso = pvalues_iso[:,:,index,:]

    postrial_pvalues = np.array([ np.sum(pvalues_iso[i, j] < pvalues_flare[i, j, :, np.newaxis], axis = 1) for i in range(pvalues_flare.shape[0]) for j in range(pvalues_flare.shape[1]) ])
    postrial_pvalues = np.reshape(postrial_pvalues, pvalues_iso.shape) / pvalues_iso.shape[2]

    #set to 1/n_samples null postrial pvalues
    min_pvalue = 1 / postrial_pvalues.shape[2]

    postrial_pvalues = ma.masked_array(postrial_pvalues, mask = (postrial_pvalues == 0)).filled(fill_value = min_pvalue)

    print('Computing postrial pvalues took', datetime.now() - start, 's')

    return postrial_pvalues

#define a function to define the style of a color bar
def create_colorbar(fig, ax, heatmap, colormap, title, limits, label_size):

    cb = fig.colorbar(heatmap, cmap = colormap, ax = ax)

    cb.ax.set_ylabel(title, fontsize=label_size)
    cb.ax.set_ylim(limits[0], limits[1])
    cb.ax.tick_params(labelsize=label_size)

#create the axis with intuitive duration for the flares
def create_intuitive_duration_axis(ax, ticks_array, labels_array):

    ax_new = ax.twiny()

    ax_new.set_xlim(ax.get_xlim())
    ax_new.set_xticks(ticks_array)
    ax_new.set_xticklabels(labels_array)

if __name__ == '__main__':

    #define a few constants
    time_begin = Time('2010-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    time_end = Time('2020-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    obs_time = time_end - time_begin
    obs_time_years = obs_time / (366*86_164)

    ra_flare = np.radians(0)
    dec_flare = np.radians(-30)

    n_events = 100_000

    #set position of the pierre auger observatory
    pao_lat = np.radians(-35.15) # this is the average latitude

    #define the maximum zenith angle
    theta_max = np.radians(80)

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

    #save the size of the target around flare
    target_radius = np.radians(float(os.path.basename(filelist_flare_info[0])[46:49]))

    #get postrial pvalues
    file_postrial_pvalues = 'Postrial_PValues_FlareLattice_patchRadius_25_targetRadius_%.1f.pkl' % np.degrees(target_radius)
    file_postrial_pvalues = os.path.join(input_path, file_postrial_pvalues)

    #clean filelists with postrial pvalues
    filelist_flare_pvalues = filelist_flare_pvalues[np.logical_not(filelist_flare_pvalues == file_postrial_pvalues)]

    #get the grid of flare intensities and durations
    with open(filelist_flare_info[0], 'rb') as file:
        flare_intensity, flare_duration = pickle.load(file)

    #compute the expected number of events in flare target
    mu_at_flare = get_mu_around_flare(ra_flare, dec_flare, target_radius, theta_max, pao_lat, n_events)

    #transform flare duration into a more meaningful quantity
    flare_duration = np.log10(flare_duration / obs_time)

    #get the merged distributions of pvalues and estimators
    estimators_flares = merge_samples(filelist_flare_estimator)
    estimators_iso = merge_samples(filelist_iso_estimator)

    pvalues_flares = merge_samples(filelist_flare_pvalues)
    pvalues_iso = merge_samples(filelist_iso_pvalues)

    #get the number of samples for each pvalue with lambda is smaller than with poisson
    performance_factor_lambda, performance_factor_lambda_corr = get_lambda_performance(pvalues_flares)
    benchmark_performance_lambda, benchmark_performance_lambda_corr = get_lambda_performance(pvalues_iso)

    #print(benchmark_performance_lambda)

    #get the postrial pvalues
    with open(file_postrial_pvalues, 'rb') as file:
        postrial_pvalues = pickle.load(file)

    postrial_pvalues_poisson = postrial_pvalues[:,:,0,:]
    postrial_pvalues_lambda = postrial_pvalues[:,:,1,:]

    #compute the fraction of postrial pvalues less than 10^(-2)
    pvalue_threshold = -2
    postrial_pvalues_poisson_below_threshold = 100*(np.sum(np.log10(postrial_pvalues_poisson) < pvalue_threshold, axis = 2) / postrial_pvalues_poisson.shape[2])
    postrial_pvalues_lambda_below_threshold = 100*(np.sum(np.log10(postrial_pvalues_lambda) < pvalue_threshold, axis = 2) / postrial_pvalues_lambda.shape[2])

    #compute the 5 % quantile of the distribution of pos trial pvalues
    quantile = .05
    quantile_postrial_pvalues_poisson = np.log10(np.quantile(postrial_pvalues_poisson, quantile, axis = 2))
    quantile_postrial_pvalues_lambda = np.log10(np.quantile(postrial_pvalues_lambda, quantile, axis = 2))

    #print(quantile_postrial_pvalues_poisson)
    #print(quantile_postrial_pvalues_lambda)

    #save colormap
    colormap = plt.get_cmap('RdBu_r')
    colormap_postrial_pv = plt.get_cmap('magma')

    #----------------------------
    # Plotting
    #----------------------------

    #initialize figure
    fig_lambda_performance_against_poisson = plt.figure(figsize=(10, 4))
    fig_postrial_pvalues = plt.figure(figsize=(10, 4))
    fig_postrial_pvs_quantile = plt.figure(figsize=(10, 4))

    postrial_pv_title = r'$\mu = %.2f$ events, $T_{\mathrm{obs}} = %.0f$ years' % (mu_at_flare, obs_time_years)

    fig_postrial_pvalues.suptitle(postrial_pv_title, fontsize = 12)
    fig_postrial_pvs_quantile.suptitle(postrial_pv_title, fontsize = 12)

    #initialize axis
    ax_lambda_performance = fig_lambda_performance_against_poisson.add_subplot(1, 2, 1)
    ax_corrected_lambda_performance = fig_lambda_performance_against_poisson.add_subplot(1, 2, 2)

    ax_postrial_pvalues_poisson = fig_postrial_pvalues.add_subplot(1, 2, 1)
    ax_postrial_pvalues_lambda = fig_postrial_pvalues.add_subplot(1, 2, 2)

    ax_postrial_pvs_quantile_poisson = fig_postrial_pvs_quantile.add_subplot(1, 2, 1)
    ax_postrial_pvs_quantile_lambda = fig_postrial_pvs_quantile.add_subplot(1, 2, 2)

    #compute the limits of the lambda performance and for the postrial pvalues
    lambda_performance_lower = 0
    lambda_performance_upper = 100
    lambda_performance_contour_levels = np.append(np.arange(lambda_performance_lower, lambda_performance_upper, 5), lambda_performance_upper)

    postrial_pvalues_lower = 0
    postrial_pvalues_upper = 100
    postrial_pvalues_contours = np.append(np.arange(postrial_pvalues_lower, postrial_pvalues_upper, 5), postrial_pvalues_upper)

    quantile_postrial_pvs_lower = -np.log10(postrial_pvalues.shape[3])
    quantile_postrial_pvs_upper = 0
    quantile_postrial_pvs_contours = np.append(np.arange(quantile_postrial_pvs_lower, quantile_postrial_pvs_upper, .25), quantile_postrial_pvs_upper)

    #plot the contours of the lambda perfomance and of postrial pvalues
    contour_lambda_performance = ax_lambda_performance.contourf(flare_duration, flare_intensity, np.transpose(performance_factor_lambda), levels = lambda_performance_contour_levels, cmap = colormap) #norm = mcolors.TwoSlopeNorm(vmin = contour_levels[0], vcenter = np.log10(1 - poisson_pvalue), vmax =contour_levels[-1]))
    contour_corrected_lambda_performance = ax_corrected_lambda_performance.contourf(flare_duration, flare_intensity, np.transpose(performance_factor_lambda_corr), levels = lambda_performance_contour_levels, cmap = colormap)

    contour_postrial_pvalues_poisson = ax_postrial_pvalues_poisson.contourf(flare_duration, flare_intensity, np.transpose(postrial_pvalues_poisson_below_threshold), levels = postrial_pvalues_contours, cmap = colormap_postrial_pv)
    contour_postrial_pvalues_lambda = ax_postrial_pvalues_lambda.contourf(flare_duration, flare_intensity, np.transpose(postrial_pvalues_lambda_below_threshold), levels = postrial_pvalues_contours, cmap = colormap_postrial_pv)

    contour_postrial_pvs_quantile_poisson = ax_postrial_pvs_quantile_poisson.contourf(flare_duration, flare_intensity, np.transpose(quantile_postrial_pvalues_poisson), levels = quantile_postrial_pvs_contours, cmap = colormap_postrial_pv)
    contour_postrial_pvs_quantile_lambda = ax_postrial_pvs_quantile_lambda.contourf(flare_duration, flare_intensity, np.transpose(quantile_postrial_pvalues_lambda), levels = quantile_postrial_pvs_contours, cmap = colormap_postrial_pv)

    #plot the contour corresponding to equal performance for lambda performance
    equal_performance_contour_lambda = ax_lambda_performance.contour(flare_duration, flare_intensity, np.transpose(performance_factor_lambda), levels = [benchmark_performance_lambda[0, 0]], colors = 'black', linestyles = 'dashed', linewidths = 1)
    equal_performance_contour_lambda_corr = ax_corrected_lambda_performance.contour(flare_duration, flare_intensity, np.transpose(performance_factor_lambda_corr), levels = [benchmark_performance_lambda_corr[0, 0]], colors = 'black', linestyles = 'dashed', linewidths = 1)

    #define the style of the axis
    lambda_performance_title = r'$\mu = %.2f$ events, $T_{\mathrm{obs}} = %.0f$ years' % (mu_at_flare, obs_time_years)
    lambda_performance_xlabel = r'$\log_{10} \left( \Delta t_{\mathrm{flare}} / 10 \mathrm{\,years} \right)$'
    lambda_performance_ylabel = r'$n_{\mathrm{events}}$'

    postrial_pv_xlabel = r'$\log_{10} \left( \Delta t_{\mathrm{flare}} / 10 \mathrm{\,years} \right)$'
    postrial_pv_ylabel = r'$n_{\mathrm{events}}$'

    ax_lambda_performance = set_style(ax_lambda_performance, lambda_performance_title, lambda_performance_xlabel, lambda_performance_ylabel, 12)
    ax_corrected_lambda_performance = set_style(ax_corrected_lambda_performance, lambda_performance_title, lambda_performance_xlabel, lambda_performance_ylabel, 12)

    ax_postrial_pvalues_poisson = set_style(ax_postrial_pvalues_poisson, 'Poisson', postrial_pv_xlabel, postrial_pv_ylabel, 12)
    ax_postrial_pvalues_lambda = set_style(ax_postrial_pvalues_lambda, r'$\Lambda$', postrial_pv_xlabel, postrial_pv_ylabel, 12)

    ax_postrial_pvs_quantile_poisson = set_style(ax_postrial_pvs_quantile_poisson, 'Poisson', postrial_pv_xlabel, postrial_pv_ylabel, 12)
    ax_postrial_pvs_quantile_lambda = set_style(ax_postrial_pvs_quantile_lambda, r'$\Lambda$', postrial_pv_xlabel, postrial_pv_ylabel, 12)

    #create and define style of color bar
    cb_title_lambda_performance = r'frac. of samples with $p_{\Lambda} \leq p_{n} \,(\%)$'
    cb_title_postrial_pv = r'frac. of samples with $p^* \leq 10^{-2} \,(\%)$'
    cb_title_postrial_pv_quantile = r'$5 \%$ quantile of $p^*$ distribution'

    create_colorbar(fig_lambda_performance_against_poisson, ax_lambda_performance, contour_lambda_performance, colormap, cb_title_lambda_performance, [lambda_performance_lower, lambda_performance_upper], 12)
    create_colorbar(fig_lambda_performance_against_poisson, ax_corrected_lambda_performance, contour_corrected_lambda_performance, colormap, cb_title_lambda_performance, [lambda_performance_lower, lambda_performance_upper], 12)

    create_colorbar(fig_postrial_pvalues, ax_postrial_pvalues_poisson, contour_postrial_pvalues_poisson, colormap_postrial_pv, cb_title_postrial_pv, [postrial_pvalues_lower, postrial_pvalues_upper], 12)
    create_colorbar(fig_postrial_pvalues, ax_postrial_pvalues_lambda, contour_postrial_pvalues_lambda, colormap_postrial_pv, cb_title_postrial_pv, [postrial_pvalues_lower, postrial_pvalues_upper], 12)

    create_colorbar(fig_postrial_pvs_quantile, ax_postrial_pvs_quantile_poisson, contour_postrial_pvs_quantile_poisson, colormap_postrial_pv, cb_title_postrial_pv_quantile, [quantile_postrial_pvs_lower, quantile_postrial_pvs_upper], 12)
    create_colorbar(fig_postrial_pvs_quantile, ax_postrial_pvs_quantile_lambda, contour_postrial_pvs_quantile_lambda, colormap_postrial_pv, cb_title_postrial_pv_quantile, [quantile_postrial_pvs_lower, quantile_postrial_pvs_upper], 12)

    #plot axis with intuitive time scales
    intuitive_duration_array = np.log10(np.array([1, 7, 30, 366]) * 86_164 / obs_time) # in seconds
    intuitive_duration_array_label = ['1 day', '1 week', '1 month', '1 year']

    create_intuitive_duration_axis(ax_lambda_performance, intuitive_duration_array, intuitive_duration_array_label)
    create_intuitive_duration_axis(ax_corrected_lambda_performance, intuitive_duration_array, intuitive_duration_array_label)
    create_intuitive_duration_axis(ax_postrial_pvalues_poisson, intuitive_duration_array, intuitive_duration_array_label)
    create_intuitive_duration_axis(ax_postrial_pvalues_lambda, intuitive_duration_array, intuitive_duration_array_label)
    create_intuitive_duration_axis(ax_postrial_pvs_quantile_poisson, intuitive_duration_array, intuitive_duration_array_label)
    create_intuitive_duration_axis(ax_postrial_pvs_quantile_lambda, intuitive_duration_array, intuitive_duration_array_label)

    #plot vertical lines with the intuitive time scales
    ax_lambda_performance.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')
    ax_corrected_lambda_performance.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')

    ax_postrial_pvalues_poisson.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')
    ax_postrial_pvalues_lambda.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')

    ax_postrial_pvs_quantile_poisson.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')
    ax_postrial_pvs_quantile_lambda.vlines(x = intuitive_duration_array, ymin = flare_intensity[0,0], ymax = flare_intensity[0,-1], color = 'tab:gray', alpha = .7, linestyle = 'dashed')

    fig_lambda_performance_against_poisson.tight_layout()
    fig_postrial_pvalues.tight_layout()
    fig_postrial_pvs_quantile.tight_layout()

    fig_lambda_performance_against_poisson.savefig(os.path.join(output_path, 'LambdaPValues_below_poissonPValue_muAtFlare_%.2f_targetRadius_%.1f.pdf' % (mu_at_flare, np.degrees(target_radius))))
    fig_postrial_pvalues.savefig(os.path.join(output_path, 'postrialPValues_below_threshold_muAtFlare_%.2f_targetRadius_%.1f.pdf' % (mu_at_flare, np.degrees(target_radius))))
    fig_postrial_pvs_quantile.savefig(os.path.join(output_path, 'postrialPValues_Quantile005_muAtFlare_%.2f_targetRadius_%.1f.pdf' % (mu_at_flare, np.degrees(target_radius))))

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
