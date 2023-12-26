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
from hist_manip import get_bin_width

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
def get_kde_lambda_dist_per_mu(index, lambda_dist_per_mu, norm):

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

    #transform the pdf into counts to use a likelihood fit
    lambda_bin_content = np.ceil(norm * lambda_bin_content).astype('int')
    lambda_bin_error = np.sqrt(lambda_bin_content)

    #compute bin errors. Mind that each lambda distribution was estimated from 100 samples of skies, so an additional factor of 100 is needed to estimate the errors
    #lambda_bin_error = np.sqrt(lambda_bin_content) / 100

    #normalize distribution
    #integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

    #lambda_bin_content = lambda_bin_content / integral
    #lambda_bin_error = lambda_bin_error / integral

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
    output_path= 'datasets/suboptimal_fit_parameters' #'./results/' + os.path.splitext(os.path.basename(sys.argv[0]))[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #save file containing distribution of lambda as a function of rate
    input_path = './datasets/lambda_dist'

    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    file_lambda_dist = 'Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_10000.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_lambda_dist = os.path.join(input_path, file_lambda_dist)

    #check if both requested file exist
    if not os.path.exists(file_lambda_dist):
        print('Requested file does not exist!')
        exit()

    #load files with the kernel density estimation of Lambda
    n_samples = 100

    file_kde_lambda_dist = 'GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_100.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_kde_lambda_dist = os.path.join(input_path, file_kde_lambda_dist)

    #save the corresponding dataframes
    kde_lambda_dist_per_mu = pd.read_json(file_kde_lambda_dist)
    lambda_dist_per_mu = pd.read_json(file_lambda_dist)

    # ------------------------------------------
    # Plot the chi2 and the likelihood as a function of the fit initial point, per mu value
    # ------------------------------------------
    quantile_lower = 0.99
    quantile_upper = 0.9999

    #initialize figure
    fig_fit_initial_point = plt.figure(figsize = (15, 4))

    fig_fit_initial_point.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)

    #initialize axis for the plots
    ax_fit_initial_chi2 = fig_fit_initial_point.add_subplot(1, 2, 1)
    ax_fit_initial_loglike = fig_fit_initial_point.add_subplot(1, 2, 2)

    #get the colormap
    colormap_fit_init = plt.get_cmap('RdBu_r')
    colormap_fit_init.set_bad(color = 'lightgray', alpha = 1)

    #produce the color array
    mu_array = np.array(lambda_dist_per_mu['mu_low_edges'].values)

    #save fit parameters as a function of the initial point for the regular and kde estimation of lambda pdf
    chi2_per_mu_per_fit_init = []
    chi2_kde_per_mu_per_fit_init = []
    chi2_slope_per_mu_per_fit_init = []
    chi2_kde_slope_per_mu_per_fit_init = []

    loglike_per_mu_per_fit_init = []
    loglike_kde_per_mu_per_fit_init = []
    loglike_slope_per_mu_per_fit_init = []
    loglike_kde_slope_per_mu_per_fit_init = []

    fit_init_arrays = []

    #fit the binned distribution of lambda using a chi2 or a loglike fit
    for i in range(mu_array.shape[0]):

        mu_lower_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error = get_lambda_dist_per_mu(i, lambda_dist_per_mu)

        #integrate the lambda dist pdf
        integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

        #get the gaussian kde lambda pdf with the correct normalization to allow a likelihood fit
        _, _, kde_lambda_bin_centers, kde_lambda_bin_content, kde_lambda_bin_error = get_kde_lambda_dist_per_mu(i, kde_lambda_dist_per_mu, integral / n_samples)

        #compute the cdf to determine the lower and upper quantiles
        cdf_lambda = np.cumsum(lambda_bin_content) / np.sum(lambda_bin_content)

        #define the initial point of the fit and fit the lambda distribution
        fit_init_lower = lambda_bin_centers[cdf_lambda > quantile_lower][0]
        fit_init_upper = lambda_bin_centers[cdf_lambda > quantile_upper][0]

        fit_init_arrays.append(lambda_bin_centers[np.logical_and( lambda_bin_centers >= fit_init_lower, lambda_bin_centers <= fit_init_upper)])

        #perform the chi2 and likelihood fit
        loglike_fit_popt_array, loglike_array = perform_likelihood_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, [fit_init_lower, fit_init_upper])
        kde_loglike_fit_popt_array, kde_loglike_array = perform_likelihood_fit_exp(kde_lambda_bin_centers, kde_lambda_bin_content, kde_lambda_bin_error, [fit_init_lower, fit_init_upper])

        chi2_fit_popt_array, chi2_array = perform_chi2_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, [fit_init_lower, fit_init_upper])
        kde_chi2_fit_popt_array, kde_chi2_array = perform_chi2_fit_exp(kde_lambda_bin_centers, kde_lambda_bin_content, kde_lambda_bin_error, [fit_init_lower, fit_init_upper])

        #save parameters
        loglike_per_mu_per_fit_init.append(loglike_array)
        loglike_kde_per_mu_per_fit_init.append(kde_loglike_array)
        loglike_slope_per_mu_per_fit_init.append(loglike_fit_popt_array[:, 1])
        loglike_kde_slope_per_mu_per_fit_init.append(kde_loglike_fit_popt_array[:, 1])

        chi2_per_mu_per_fit_init.append(chi2_array)
        chi2_kde_per_mu_per_fit_init.append(kde_chi2_array)
        chi2_slope_per_mu_per_fit_init.append(chi2_fit_popt_array[:, 1])
        chi2_kde_slope_per_mu_per_fit_init.append(kde_chi2_fit_popt_array[:, 1])

    #prepare arrays for plotting
    #loglike
    fit_init_array, loglike_per_mu_per_fit_init = prepare_arrays_for_plotting(loglike_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, loglike_kde_per_mu_per_fit_init = prepare_arrays_for_plotting(loglike_kde_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, loglike_slope_per_mu_per_fit_init = prepare_arrays_for_plotting(loglike_slope_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, loglike_kde_slope_per_mu_per_fit_init = prepare_arrays_for_plotting(loglike_kde_slope_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)

    #chi2 fit
    _, chi2_per_mu_per_fit_init = prepare_arrays_for_plotting(chi2_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, chi2_kde_per_mu_per_fit_init = prepare_arrays_for_plotting(chi2_kde_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, chi2_slope_per_mu_per_fit_init = prepare_arrays_for_plotting(chi2_slope_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)
    _, chi2_kde_slope_per_mu_per_fit_init = prepare_arrays_for_plotting(chi2_kde_slope_per_mu_per_fit_init, fit_init_arrays, lambda_bin_centers)

    #define the names of the output files
    output_loglike = 'LogLikelihood_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'
    output_kde_loglike = 'GKDE_LogLikelihood_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'
    output_chi2 = 'Chi2_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'
    output_kde_chi2 = 'GKDE_Chi2_fit_LambdaDist_fitParameters_movingInitialPoint.pkl'

    output_loglike = os.path.join(output_path, output_loglike)
    output_kde_loglike = os.path.join(output_path, output_kde_loglike)
    output_chi2 = os.path.join(output_path, output_chi2)
    output_kde_chi2 = os.path.join(output_path, output_kde_chi2)

    #save pickled files with the parameters
    with open(output_loglike, 'wb') as file:
        pickle.dump(( mu_array, fit_init_array, loglike_per_mu_per_fit_init , loglike_slope_per_mu_per_fit_init ), file)

    with open(output_kde_loglike, 'wb') as file:
        pickle.dump(( mu_array, fit_init_array, loglike_kde_per_mu_per_fit_init , loglike_kde_slope_per_mu_per_fit_init ), file)

    with open(output_chi2, 'wb') as file:
        pickle.dump(( mu_array, fit_init_array, chi2_per_mu_per_fit_init , chi2_slope_per_mu_per_fit_init ), file)

    with open(output_kde_chi2, 'wb') as file:
        pickle.dump(( mu_array, fit_init_array, chi2_kde_per_mu_per_fit_init , chi2_kde_slope_per_mu_per_fit_init ), file)
