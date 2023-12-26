import pandas as pd
import numpy as np
import healpy as hp
from healpy.newvisufunc import projview

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation

import pickle
import os
import sys

import scipy.interpolate as spline
from scipy.special import gamma

sys.path.append('../src/')

import hist_manip
from fit_routines import perform_chi2_fit_exp
from fit_routines import perform_likelihood_fit_exp
from xmax_fit_routines import perform_fit_gumble

from hist_manip import data_2_binned_errorbar
from hist_manip import get_bin_centers

from event_manip import compute_directional_exposure
from event_manip import time_ordered_events
from event_manip import compute_lambda_correction
from event_manip import ang_diff
from event_manip import get_normalized_exposure_map
from event_manip import get_skymap

from axis_style import set_style

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#compute the cdf of the lambda distribution for each bin, and include the fit parameters
def save_lambda_cdf(output_path, file_kde_lambda_dist, fit_init, tail_slope):

    #save the dataframe
    cdf_lambda_dist = pd.read_json(file_kde_lambda_dist)

    #define the column names
    #column_names = ['mu_low_edges', 'mu_upper_edges', 'lambda_bin_centers', 'cdf_lambda_bin_content', 'fit_init', 'tail_slope']

    cdf_lambda_dist['cdf_lambda_bin_content'] = cdf_lambda_dist['lambda_bin_content'].apply(lambda x: np.cumsum(np.array(x)))
    cdf_lambda_dist['fit_init'] = pd.Series(fit_init)
    cdf_lambda_dist['tail_slope'] = pd.Series(tail_slope)

    print(cdf_lambda_dist)

    #define the name of the file
    cdf_lambda_output = 'CDF_' + os.path.basename(file_kde_lambda_dist)

    cdf_lambda_dist.to_json(os.path.join(output_path, cdf_lambda_output), index = True)


#get the distribution of lambda for a given value of expected number of events
def get_kde_lambda_dist_per_mu(index, lambda_dist_per_mu, n_samples):

    #save correspoding value of mu
    mu_low_edge = lambda_dist_per_mu['mu_low_edges'].loc[index]
    mu_upper_edge = lambda_dist_per_mu['mu_upper_edges'].loc[index]

    #save the bins edges of the lambda distribution
    lambda_bin_edges = np.array(lambda_dist_per_mu['lambda_bin_edges'].loc[index])
    lambda_bin_content = np.array(lambda_dist_per_mu['lambda_bin_content'].loc[index])#this must be edited!!!!

    #compute bin centers
    lambda_bin_centers = get_bin_centers(lambda_bin_edges)

    #clean values of the pdf that are very close to 0
    is_zero = np.isclose(lambda_bin_content, 0, atol = 1e-7)

    #lambda_bin_centers[is_zero] = 0 #lambda_bin_centers[np.logical_not(is_zero)]
    lambda_bin_content[is_zero] = 0 #lambda_bin_content[np.logical_not(is_zero)]

    #compute bin errors. Mind that each lambda distribution was estimated from 100 samples of skies, so an additional factor of 100 is needed to estimate the errors
    lambda_bin_error = np.sqrt(lambda_bin_content) / 100

    #normalize distribution
    integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

    lambda_bin_content = lambda_bin_content / integral
    lambda_bin_error = lambda_bin_error / integral

    return mu_low_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error

#get the distribution of lambda for a given value of expected number of events
def get_lambda_dist_per_mu(index, lambda_dist_per_mu):

    #save correspoding value of mu
    mu_low_edge = lambda_dist_per_mu['mu_low_edges'].loc[index]
    mu_upper_edge = lambda_dist_per_mu['mu_upper_edges'].loc[index]

    #save the bins edges of the lambda distribution
    lambda_bin_edges = np.array(lambda_dist_per_mu['lambda_bin_edges'].loc[index])
    lambda_bin_content = np.array(lambda_dist_per_mu['lambda_bin_content'].loc[index])#this must be edited!!!!

    #compute bin centers
    lambda_bin_centers = get_bin_centers(lambda_bin_edges)

    #clean values of the pdf that are very close to 0
    #is_zero = np.isclose(lambda_bin_content, 0, atol = 1e-7)

    #lambda_bin_centers[is_zero] = 0 #lambda_bin_centers[np.logical_not(is_zero)]
    #lambda_bin_content[is_zero] = 0 #lambda_bin_content[np.logical_not(is_zero)]

    #compute bin errors. Mind that each lambda distribution was estimated from 100 samples of skies, so an additional factor of 100 is needed to estimate the errors
    lambda_bin_error = np.sqrt(lambda_bin_content) #/ 100

    #normalize distribution
    #integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

    #lambda_bin_content = lambda_bin_content / integral
    #lambda_bin_error = lambda_bin_error / integral

    return mu_low_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error

#define a function to define the style of a color bar
def create_colorbar(fig, ax, colormap, title, limits, label_size):

    cb = fig.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=limits[0], vmax=limits[1]), cmap = colormap), ax = ax)

    cb.ax.set_ylabel(title, fontsize=label_size)
    cb.ax.set_ylim(limits[0], limits[1])
    cb.ax.tick_params(labelsize=label_size)

#compute the deviance between the fitted and binned lambda distribution
def get_fit_deviance(lambda_bin_centers, lambda_bin_content, lambda_bin_error, fitted_lambda_dist):

    #converts lists into arrays
    lambda_bin_centers = np.array(lambda_bin_centers)
    lambda_bin_content = np.array(lambda_bin_content)

    #save fit parameters
    const = fitted_lambda_dist[0][0]
    tail_slope = fitted_lambda_dist[0][1]
    fit_init = fitted_lambda_dist[2][0]

    #define the fit function and computes at the bin centers
    fit_bin_content = const*np.exp( -lambda_bin_centers * tail_slope )

    #compute the deviance
    deviance = np.empty(lambda_bin_centers.shape)

    are_not_defined = lambda_bin_centers < fit_init #np.logical_or(lambda_bin_centers < fit_init) , lambda_bin_content == 0)
    are_defined = np.logical_not(are_not_defined)

    deviance[are_not_defined] = np.nan
    deviance[are_defined] = (fit_bin_content[are_defined] - lambda_bin_content[are_defined]) / lambda_bin_error[are_defined]

    return deviance

#prepares the arrays with the sucession of initial fit points and values of goodness of fit
def prepare_arrays_for_plotting(loglike_arrays, fit_init_arrays, lambda_bin_centers):

    #save array of min (max) values of each fit range
    min_array = np.array([np.min(array) for array in fit_init_arrays])
    max_array = np.array([np.max(array) for array in fit_init_arrays])

    #compute the absolute minimum and max of the initial fit points
    min_fit_init = np.min(min_array)
    max_fit_init = np.max(max_array)

    #build the fit init array
    fit_init_array = lambda_bin_centers[np.logical_and(lambda_bin_centers >= min_fit_init, lambda_bin_centers <= max_fit_init)]

    #compute the positions of the min (max) of each fit range with respect to the absolute min and max
    min_indices_array = np.searchsorted(fit_init_array, min_array)
    max_indices_array = np.searchsorted(fit_init_array, max_array)

    #save the length of the final fit init array
    fit_initial_size = fit_init_array.shape[0]
    mu_size = len(loglike_arrays)

    #initialize array with nan values
    loglike_grid = np.full((mu_size, fit_initial_size), np.nan)

    #fill the values of lohlike in the respective positions
    for i, array in enumerate(loglike_arrays):

        lower_index = min_indices_array[i]
        upper_index = max_indices_array[i] + 1

        loglike_grid[i, lower_index:upper_index] = array

    return fit_init_array, loglike_grid

#define the main function
if __name__ == '__main__':

    #save name of output path and creates it is does not exist
    output_path= './results/' + os.path.splitext(os.path.basename(sys.argv[0]))[0][5:]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #save file containing distribution of lambda as a function of rate
    input_path = './datasets/suboptimal_fit_parameters'

    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    #load files with the fit results
    file_loglike_fit = 'LogLikelihood_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'
    file_kde_loglike_fit = 'GKDE_LogLikelihood_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'

    file_chi2_fit = 'Chi2_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'
    file_kde_chi2_fit = 'GKDE_Chi2_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'

    file_loglike_fit = os.path.join(input_path, file_loglike_fit)
    file_kde_loglike_fit = os.path.join(input_path, file_kde_loglike_fit)

    file_chi2_fit = os.path.join(input_path, file_chi2_fit)
    file_kde_chi2_fit = os.path.join(input_path, file_kde_chi2_fit)

    #save objects from pickled files
    with open(file_loglike_fit, 'rb') as file:
        mu_array, fit_init_array, loglike_values, loglike_slopes = pickle.load(file)

    with open(file_chi2_fit, 'rb') as file:
        _, _, chi2_values, chi2_slopes = pickle.load(file)

    with open(file_kde_loglike_fit, 'rb') as file:
        _, _, kde_loglike_values, kde_loglike_slopes = pickle.load(file)

    with open(file_kde_chi2_fit, 'rb') as file:
        _, _, kde_chi2_values, kde_chi2_slopes = pickle.load(file)

    #compute the averaged slope and dispersion according to the initial point of the fit
    mean_slope_loglike = np.nanmean(loglike_slopes, axis = 1)
    mean_slope_kde_loglike = np.nanmean(kde_loglike_slopes, axis = 1)
    mean_slope_chi2 = np.nanmean(chi2_slopes, axis = 1)
    mean_slope_kde_chi2 = np.nanmean(kde_chi2_slopes, axis = 1)

    sigma_slope_loglike = np.nanstd(loglike_slopes, axis = 1)
    sigma_slope_kde_loglike = np.nanstd(kde_loglike_slopes, axis = 1)
    sigma_slope_chi2 = np.nanstd(chi2_slopes, axis = 1)
    sigma_slope_kde_chi2 = np.nanstd(kde_chi2_slopes, axis = 1)

    amplitude_slope_loglike = np.nanmax(loglike_slopes, axis = 1) - np.nanmin(loglike_slopes, axis = 1)
    amplitude_slope_kde_loglike = np.nanmax(kde_loglike_slopes, axis = 1) - np.nanmin(kde_loglike_slopes, axis = 1)
    amplitude_slope_chi2 = np.nanmax(chi2_slopes, axis = 1) - np.nanmin(chi2_slopes, axis = 1)
    amplitude_slope_kde_chi2 = np.nanmax(kde_chi2_slopes, axis = 1) - np.nanmin(kde_chi2_slopes, axis = 1)

    #compute residuals of determination of the slope
    slope_residuals_chi2 = 100 * ( 1- (chi2_slopes / mean_slope_chi2[:, np.newaxis]))
    slope_residuals_kde_chi2 = 100 * ( 1- (kde_chi2_slopes / mean_slope_kde_chi2[:, np.newaxis]))

    slope_residuals_loglike = 100 * ( 1 - (loglike_slopes / mean_slope_loglike[:, np.newaxis]))
    slope_residuals_kde_loglike = 100 * ( 1 - (kde_loglike_slopes / mean_slope_kde_loglike[:, np.newaxis]))

    # ------------------------------------------
    # Plot the chi2 and the likelihood as a function of the fit initial point, per mu value
    # ------------------------------------------
    #initialize figure
    fig_gof_per_mu_per_fit_init = plt.figure(figsize = (10, 4))
    fig_slope_per_mu_per_fit_init = plt.figure(figsize = (15, 4))

    fig_gof_per_mu_per_fit_init.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)
    fig_slope_per_mu_per_fit_init.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)

    #initialize axis for the plots
    ax_gof_chi2 = fig_gof_per_mu_per_fit_init.add_subplot(1, 2, 1)
    ax_gof_loglike = fig_gof_per_mu_per_fit_init.add_subplot(1, 2, 2)

    ax_slope_chi2 = fig_slope_per_mu_per_fit_init.add_subplot(1, 3, 1)
    ax_slope_loglike = fig_slope_per_mu_per_fit_init.add_subplot(1, 3, 2)
    ax_mean_slope = fig_slope_per_mu_per_fit_init.add_subplot(1, 3, 3)

    #get the colormap
    colormap_gof = plt.get_cmap('RdBu_r')
    colormap_slope = plt.get_cmap('magma')

    colormap_gof.set_bad(color = 'lightgray', alpha = 1)
    colormap_slope.set_bad(color = 'lightgray', alpha = 1)

    #plot heatmap corresponding to the values of the goodness of fit and tail slopes for chi2 and likelihood fits
    loglike_lower = 0
    loglike_upper = 2

    chi2_lower = 0
    chi2_upper = 5

    slope_lower = -20
    slope_upper = 20

    heatmap_loglike = ax_gof_loglike.pcolormesh(fit_init_array, mu_array, loglike_values, vmin = loglike_lower, vmax = loglike_upper, cmap = colormap_gof, edgecolors = 'face')
    heatmap_chi2 = ax_gof_chi2.pcolormesh(fit_init_array, mu_array, chi2_values, vmin = chi2_lower, vmax = chi2_upper, cmap = colormap_gof, edgecolors = 'face')

    heatmap_slope_loglike = ax_slope_loglike.pcolormesh(fit_init_array, mu_array, slope_residuals_loglike, vmin = slope_lower, vmax = slope_upper, cmap = colormap_gof, edgecolors = 'face')
    heatmap_slope_chi2 = ax_slope_chi2.pcolormesh(fit_init_array, mu_array, slope_residuals_chi2, vmin = slope_lower, vmax = slope_upper, cmap = colormap_gof, edgecolors = 'face')

    #plot the mean slope as a function of the mean number of events
    ax_mean_slope.errorbar(mu_array, mean_slope_chi2, yerr = sigma_slope_chi2, color = 'tab:blue', marker = 'o', markersize = 3, linestyle = 'None', label = '$\chi^2$ fit')
    ax_mean_slope.errorbar(mu_array, mean_slope_loglike, yerr = sigma_slope_loglike, color = 'tab:red', marker = 'o', markersize = 3, linestyle = 'None', label = 'Likelihood fit')

    #plot colorbars
    create_colorbar(fig_gof_per_mu_per_fit_init, ax_gof_loglike, colormap_gof, r'$\ln \mathcal{L}$', [loglike_lower, loglike_upper], 14)
    create_colorbar(fig_gof_per_mu_per_fit_init, ax_gof_chi2, colormap_gof, r'$\chi^2 / \mathrm{ndf}$', [chi2_lower, chi2_upper], 14)

    create_colorbar(fig_slope_per_mu_per_fit_init, ax_slope_loglike, colormap_gof, r'$1 - \frac{\beta}{\langle \beta \rangle}$', [slope_lower, slope_upper], 14)
    create_colorbar(fig_slope_per_mu_per_fit_init, ax_slope_chi2, colormap_gof, r'$1 - \frac{\beta}{\langle \beta \rangle}$', [slope_lower, slope_upper], 14)

    #define the style of the axis
    ax_gof_loglike = set_style(ax_gof_loglike, 'Likelihood fit', r'$\Lambda_0$', r'$\mu$', 14)
    ax_gof_chi2 = set_style(ax_gof_chi2, '$\chi^2$ fit', r'$\Lambda_0$', r'$\mu$', 14)

    ax_slope_loglike = set_style(ax_slope_loglike, 'Likelihood fit', r'$\Lambda_0$', r'$\mu$', 14)
    ax_slope_chi2 = set_style(ax_slope_chi2, '$\chi^2$ fit', r'$\Lambda_0$', r'$\mu$', 14)
    ax_mean_slope = set_style(ax_mean_slope, '', r'$\mu$', r'$\langle \beta \rangle$', 14)

    #plot legends
    ax_mean_slope.legend(loc = 'lower center', fontsize = 14)

    #save figure
    fig_gof_per_mu_per_fit_init.tight_layout()
    fig_slope_per_mu_per_fit_init.tight_layout()

    fig_gof_per_mu_per_fit_init.savefig(os.path.join(output_path, 'LambdaDist_optimizing_initial_point_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))
    fig_slope_per_mu_per_fit_init.savefig(os.path.join(output_path, 'LambdaDist_Slope_optimizing_initial_point_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))

    # ------------------------------------------
    # Compare results for binned and kernel density estimated Lambda distributions
    # ------------------------------------------
    #initialize figure
    fig_binned_vs_kde_slope = plt.figure(figsize = (15, 4))

    fig_binned_vs_kde_slope.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)

    #initialize axis for the plots
    ax_slope_binned_lambda_dist = fig_binned_vs_kde_slope.add_subplot(1, 3, 1)
    ax_slope_kde_lambda_dist = fig_binned_vs_kde_slope.add_subplot(1, 3, 2)
    ax_mean_slope_kde = fig_binned_vs_kde_slope.add_subplot(1, 3, 3)

    #plot heatmap corresponding to the values of the goodness of fit and tail slopes for chi2 and likelihood fits
    new_slope_lower = -5
    new_slope_upper = 5

    heatmap_slope_loglike_binned = ax_slope_binned_lambda_dist.pcolormesh(fit_init_array, mu_array, slope_residuals_loglike, vmin = new_slope_lower , vmax = new_slope_upper, cmap = colormap_gof, edgecolors = 'face')
    heatmap_slope_loglike_kde = ax_slope_kde_lambda_dist.pcolormesh(fit_init_array, mu_array, slope_residuals_kde_loglike, vmin = new_slope_lower, vmax = new_slope_upper, cmap = colormap_gof, edgecolors = 'face')

    #plot the mean slope as a function of the mean number of events
    ax_mean_slope_kde.errorbar(mu_array, mean_slope_loglike, yerr = .5*amplitude_slope_loglike, color = 'tab:blue', marker = 'o', markersize = 3, linestyle = 'None', elinewidth = 0, capsize = 5, capthick = 2, label = r'Binned $\Lambda$-dist')
    ax_mean_slope_kde.errorbar(mu_array, mean_slope_kde_loglike, yerr = .5*amplitude_slope_kde_loglike, color = 'tab:red', marker = 'o', markersize = 5, mfc = 'white', linestyle = 'None', label = r'GKDE $\Lambda$-dist', )

    #plot colorbars
    create_colorbar(fig_binned_vs_kde_slope, ax_slope_binned_lambda_dist, colormap_gof, r'$1 - \frac{\beta}{\langle \beta \rangle}\,(\%)$', [new_slope_lower, new_slope_upper], 14)
    create_colorbar(fig_binned_vs_kde_slope, ax_slope_kde_lambda_dist, colormap_gof, r'$1 - \frac{\beta}{\langle \beta \rangle}\,(\%)$', [new_slope_lower, new_slope_upper], 14)

    #define the style of the axis
    ax_slope_binned_lambda_dist = set_style(ax_slope_binned_lambda_dist, 'Binned $\Lambda$-dist', r'$\Lambda_0$', r'$\mu$', 14)
    ax_slope_kde_lambda_dist = set_style(ax_slope_kde_lambda_dist, 'GKDE $\Lambda$-dist', r'$\Lambda_0$', r'$\mu$', 14)
    ax_mean_slope_kde = set_style(ax_mean_slope_kde, '', r'$\mu$', r'$\langle \beta \rangle$', 14)

    #plot legends
    ax_mean_slope_kde.legend(loc = 'lower center', fontsize = 14)

    #save figure
    fig_binned_vs_kde_slope.tight_layout()

    fig_binned_vs_kde_slope.savefig(os.path.join(output_path, 'Binned_vs_GKDE_LambdaDist_Slope_optimizing_initial_point_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))
