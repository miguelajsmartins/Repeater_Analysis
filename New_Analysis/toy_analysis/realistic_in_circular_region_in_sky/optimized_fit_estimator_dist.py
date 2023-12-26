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
#from optimized_fit_routines import perform_fit_exp
from optimized_fit_routines import perform_likelihood_fit_exp

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
# def get_lambda_cdf(output_path, file_kde_lambda_dist, fit_init, tail_slope):
#
#     #save the dataframe
#     cdf_lambda_dist = pd.read_json(file_kde_lambda_dist)
#
#     #define the column names
#     #column_names = ['mu_low_edges', 'mu_upper_edges', 'lambda_bin_centers', 'cdf_lambda_bin_content', 'fit_init', 'tail_slope']
#
#     cdf_lambda_dist['cdf_lambda_bin_content'] = cdf_lambda_dist['lambda_bin_content'].apply(lambda x: np.cumsum(np.array(x)))
#     cdf_lambda_dist['fit_init'] = pd.Series(fit_init)
#     cdf_lambda_dist['tail_slope'] = pd.Series(tail_slope)
#
#     print(cdf_lambda_dist)
#
#     #define the name of the file
#     cdf_lambda_output = 'CDF_' + os.path.basename(file_kde_lambda_dist)
#
#     cdf_lambda_dist.to_json(os.path.join(output_path, cdf_lambda_output), index = True)

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

    file_lambda_dist = 'Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_10000.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_corrected_lambda_dist = 'Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_10000.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_lambda_dist = os.path.join(input_path, file_lambda_dist)
    file_corrected_lambda_dist = os.path.join(input_path, file_corrected_lambda_dist)

    #check if both requested file exist
    if (not os.path.exists(file_lambda_dist)) or (not os.path.exists(file_lambda_dist)):
        print('One of the requested files does not exist!')
        exit()

    #load files with the kernel density estimation of Lambda
    n_samples = 100
    file_kde_lambda_dist = 'GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_%i.json' % (np.degrees(patch_radius), np.degrees(target_radius), n_samples)
    file_kde_corrected_lambda_dist = 'GaussianKernelEstimated_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_%i.json' % (np.degrees(patch_radius), np.degrees(target_radius), n_samples)

    file_kde_lambda_dist = os.path.join(input_path, file_kde_lambda_dist)
    file_kde_corrected_lambda_dist = os.path.join(input_path, file_kde_corrected_lambda_dist)

    #save the corresponding dataframes
    kde_lambda_dist_per_mu = pd.read_json(file_kde_lambda_dist)
    #corrected_lambda_dist_per_mu = pd.read_json(file_kde_corrected_lambda_dist)

    lambda_dist_per_mu = pd.read_json(file_lambda_dist)
    #corrected_lambda_dist_per_mu = pd.read_json(file_kde_corrected_lambda_dist)


    # ------------------------------------------
    # Plot the distribution of Lambda and corrected Lambda for each mu value. Fits the distribution as well
    # ------------------------------------------
    quantile = 0.99

    #initialize figure
    fig_lambda_dist = plt.figure(figsize = (15, 5))
    fig_lambda_dist_correction = plt.figure(figsize = (5, 4))

    fig_lambda_dist.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)
    fig_lambda_dist_correction.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)

    ax_lambda_dist = plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=3, fig = fig_lambda_dist)
    ax_lambda_fit_deviance = plt.subplot2grid((3, 3), (0, 1), colspan=1, rowspan=3, fig = fig_lambda_dist)
    #ax_kde_lambda_dist = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1, fig = fig_lambda_dist)
    #ax_kde_corrected_lambda_dist = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_beta_mu_func = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_chi2_mu_func = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_fit_init_mu_func = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)

    ax_mean_mu_func = fig_lambda_dist_correction.add_subplot(1, 1, 1) #plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)

    #get the colormap
    colormap = plt.get_cmap('magma')
    colormap_deviance = plt.get_cmap('RdBu_r')
    colormap_deviance.set_bad(color = 'lightgray', alpha = 1)

    #produce the color array
    mu_array = np.array(lambda_dist_per_mu['mu_low_edges'].values)
    mu_edges = np.append(mu_array, np.array(lambda_dist_per_mu['mu_upper_edges'].values)[-1])
    color_array = (lambda_dist_per_mu['mu_low_edges'].values - lambda_dist_per_mu['mu_low_edges'].min()) / (lambda_dist_per_mu['mu_low_edges'].max() - lambda_dist_per_mu['mu_low_edges'].min())
    color_array = colormap(color_array)

    #save moments of distributions
    mean_lambda = []
    sigma_lambda = []
    #mean_corrected_lambda = []
    #sigma_corrected_lambda = []

    #save fit parameters
    beta_lambda = []
    beta_lambda_error = []
    beta_kde_lambda = []
    beta_kde_lambda_error = []

    #save goodness of fit
    loglike_lambda_fit = []

    chi2_lambda_fit = []
    chi2_kde_lambda_fit = []

    #beta_corrected_lambda_error = []

    #initialize lists
    deviance_lambda_fit_dist = []
    #list_lambda_bin_centers = []

    #save the initial point of the fit
    fit_init_lambda = []
    fit_init_kde_lambda = []

    #save the 99 and 99.9 quantiles
    quantile_99_mu_func = []
    quantile_999_mu_func = []

    #plot the distribution of lambda per expected value of events
    for i in range(len(lambda_dist_per_mu)):

        mu_low_edge, mu_upper_edge,lambda_bin_centers, lambda_bin_content, lambda_bin_error = get_lambda_dist_per_mu(i, lambda_dist_per_mu)

        #lambda_bin_error = lambda_bin_error / 100
        integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

        _, _, kde_lambda_bin_centers, kde_lambda_bin_content, kde_lambda_bin_error = get_kde_lambda_dist_per_mu(i, kde_lambda_dist_per_mu, integral / n_samples)

        #define the initial point of the fit and fit the lambda distribution
        cdf_lambda = np.cumsum(lambda_bin_content) / np.sum(lambda_bin_content)

        #define the initial point of the fit
        fit_initial = lambda_bin_centers[cdf_lambda > quantile][0]

        #save the 99 and the 999 quantiles
        quantile_99_mu_func.append(lambda_bin_centers[cdf_lambda > .99][0])
        quantile_999_mu_func.append(lambda_bin_centers[cdf_lambda > .999][0])


        loglike_fitted_kde_lambda_dist = perform_likelihood_fit_exp(kde_lambda_bin_centers, kde_lambda_bin_content, kde_lambda_bin_error, fit_initial)
        loglike_fitted_lambda_dist = perform_likelihood_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, fit_initial)

        #fill arrays
        mean = np.sum(kde_lambda_bin_centers*kde_lambda_bin_content) / np.sum(kde_lambda_bin_content)
        second_moment = np.sum((kde_lambda_bin_centers**2)*kde_lambda_bin_content) / np.sum(kde_lambda_bin_content)
        sigma = np.sqrt(second_moment - mean**2)

        mean_lambda.append(mean)
        sigma_lambda.append(sigma)

        beta_lambda.append(loglike_fitted_lambda_dist[0][1])
        beta_lambda_error.append(loglike_fitted_lambda_dist[1][1])
        beta_kde_lambda.append(loglike_fitted_kde_lambda_dist[0][1])
        beta_kde_lambda_error.append(loglike_fitted_kde_lambda_dist[1][1])

        #save the values of chi2 / ndf
        chi2_lambda_fit.append(loglike_fitted_lambda_dist[4])
        chi2_kde_lambda_fit.append(loglike_fitted_kde_lambda_dist[4])

        #save the values of the fit initial
        fit_init_kde_lambda.append(loglike_fitted_kde_lambda_dist[2][0])
        fit_init_lambda.append(loglike_fitted_lambda_dist[2][0])

        #compute the deviance between the fitted and lambda distributions
        fit_deviance = get_fit_deviance(lambda_bin_centers, lambda_bin_content, lambda_bin_error, loglike_fitted_lambda_dist)
        deviance_lambda_fit_dist.append(fit_deviance)

        #plot distribution for some cases
        if mu_low_edge in [23, 30]:

            ax_lambda_dist.errorbar(kde_lambda_bin_centers[::30], kde_lambda_bin_content[::30] / integral, yerr = kde_lambda_bin_error[::30] / integral, color = color_array[i], marker = 'o', linestyle = 'None', linewidth = 1, markersize = 4, mfc = 'white')
            ax_lambda_dist.errorbar(lambda_bin_centers[::30], lambda_bin_content[::30] / integral, yerr = lambda_bin_error[::30] / integral, color = color_array[i], marker = 'o', linestyle = 'None', linewidth = 1, markersize = 3)

            ax_lambda_dist.plot(loglike_fitted_kde_lambda_dist[2], loglike_fitted_kde_lambda_dist[3] / integral, color = color_array[i], linestyle = 'dashed')
            ax_lambda_dist.plot(loglike_fitted_lambda_dist[2], loglike_fitted_lambda_dist[3] / integral, color = color_array[i])


    #tranform deviance list into array
    deviance_lambda_fit_dist = np.array(deviance_lambda_fit_dist)

    #plot heatmap corresponding to the fit deviance
    deviance_lower = -2
    deviance_upper = 2

    heatmap_fit_deviance = ax_lambda_fit_deviance.pcolormesh(kde_lambda_bin_centers, mu_array, deviance_lambda_fit_dist, vmin = deviance_lower, vmax = deviance_upper, cmap = colormap_deviance, edgecolors = 'face')
    #heatmap_fit_deviance.set_cmap(colormap_deviance)

    #plot moments as a function of mean number of events
    ax_mean_mu_func.scatter(sigma_lambda, mean_lambda, edgecolors = color_array, marker = 'o', linestyle = 'None', s = 10, facecolor = 'lightgrey')
    ax_mean_mu_func.scatter(sigma_lambda, mean_lambda - compute_lambda_correction(mu_array, mu_array), c = color_array, marker = 'o', linestyle = 'None', s = 7)

    ax_beta_mu_func.errorbar(mu_array, beta_kde_lambda, yerr = beta_kde_lambda_error, color = 'tab:red', marker = 'o', linestyle = 'None', markersize = 4, mfc = 'white')
    ax_beta_mu_func.errorbar(mu_array, beta_lambda, yerr = beta_lambda_error, color = 'tab:blue', marker = 'o', linestyle = 'None', markersize = 3)

    ax_chi2_mu_func.plot(mu_array, chi2_kde_lambda_fit, color = 'tab:red', marker = 'o', linestyle = 'None', markersize = 4, mfc = 'white')
    ax_chi2_mu_func.plot(mu_array, chi2_lambda_fit, color = 'tab:blue', marker = 'o', linestyle = 'None', markersize = 3)

    ax_fit_init_mu_func.plot(mu_array, quantile_99_mu_func, color = 'grey', linestyle = 'dotted')
    #ax_fit_init_mu_func.plot(mu_array, quantile_999_mu_func, color = 'tab:gray', linestyle = 'dashed')
    ax_fit_init_mu_func.plot(mu_array, fit_init_kde_lambda, color = 'tab:red', marker = 'o', linestyle = 'None', markersize = 4, mfc = 'white')
    ax_fit_init_mu_func.plot(mu_array, fit_init_lambda, color = 'tab:blue', marker = 'o', linestyle = 'None', markersize = 3)

    #define the style of the plot
    ax_lambda_dist.set_yscale('log')
    ax_lambda_dist.set_xlim(-30, 100)
    ax_lambda_dist.set_ylim(1e-7, 1)

    ax_lambda_fit_deviance.set_xlim(30, 100)

    ax_mean_mu_func.set_ylim(-1, 20)
    ax_beta_mu_func.set_ylim(0.15, 0.3)

    #plot colorbars
    create_colorbar(fig_lambda_dist, ax_lambda_dist, colormap, r'$\mu$', [mu_array.min(), mu_array.max()], 14)
    create_colorbar(fig_lambda_dist, ax_lambda_fit_deviance, colormap_deviance, r'Fit deviance $(\sigma)$', [deviance_lower, deviance_upper], 14)
    create_colorbar(fig_lambda_dist, ax_mean_mu_func, colormap, r'$\mu$', [mu_array.min(), mu_array.max()], 14)

    #define the style of the axis
    ax_lambda_dist = set_style(ax_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)
    ax_lambda_fit_deviance = set_style(ax_lambda_fit_deviance, '', r'$\Lambda$', r'$\mu$', 14)
    ax_beta_mu_func = set_style(ax_beta_mu_func, '', '', r'$\beta$', 14)
    ax_chi2_mu_func = set_style(ax_chi2_mu_func, '', '', r'$\chi^2 / \mathrm{ndf}$', 14)
    ax_fit_init_mu_func = set_style(ax_fit_init_mu_func, '', r'$\mu$', 'Fit lower limit', 14)

    #ax_kde_lambda_dist = set_style(ax_kde_lambda_dist, '', r'$\Lambda$', r'$1 - \frac{\hat{f}_{\Lambda}(\Lambda)}{f_{\Lambda}(\Lambda)} (\%)$', 14)
    #ax_kde_corrected_lambda_dist = set_style(ax_kde_corrected_lambda_dist, '', r'$\Lambda$', r'$1 - \frac{\hat{f}_{\Lambda}(\Lambda)}{f_{\Lambda}(\Lambda)} (\%)$', 14)
    ax_mean_mu_func = set_style(ax_mean_mu_func, '', r'$\sigma(\Lambda)$', r'$\langle \Lambda \rangle$', 14)


    #ax_lambda_fit_deviance.set_facecolor('lightgray')
    ax_mean_mu_func.set_facecolor('lightgray')


    #save figure
    fig_lambda_dist.tight_layout()
    fig_lambda_dist_correction.tight_layout()

    fig_lambda_dist.savefig(os.path.join(output_path, 'Lambda_distribution_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))
    fig_lambda_dist_correction.savefig(os.path.join(output_path, 'Lambda_distribution_MeanCorr_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))

    #save the cdfs in the corresponding files
    save_lambda_cdf(input_path, file_kde_lambda_dist, fit_init_kde_lambda, beta_kde_lambda)
    save_lambda_cdf(input_path, file_lambda_dist, fit_init_lambda, beta_lambda)

    # column_names = ['mu_low_edges', 'mu_upper_edges', 'lambda_bin_centers', 'cdf_lambda_bin_content', 'fit_init', 'tail_slope']
    #
    # cdf_lambda_dist = pd.read_json(file_kde_lambda_dist)
    # cdf_corrected_lambda_dist = pd.read_json(file_kde_lambda_dist)
    #
    # cdf_lambda_dist['cdf_lambda_bin_content'] = cdf_lambda_dist['lambda_bin_content'].apply(lambda x: np.cumsum(np.array(x)))
    # cdf_lambda_dist[['fit_init', 'tail_slope']] =
    # cdf_lambda_dist['cdf_lambda_bin_content'] = cdf_lambda_dist['lambda_bin_content'].apply(lambda x: np.cumsum(np.array(x)))
    #
    # cdf_lambda_output = 'CDF_' + os.path.basename(file_lambda_dist)
    # cdf_corrected_lambda_output = 'CDF_' + os.path.basename(file_corrected_lambda_dist)
    #
    # cdf_lambda_dist.to_json(os.path.join(input_path, cdf_lambda_output), index = True)
    # cdf_corrected_lambda_dist.to_json(os.path.join(input_path, cdf_corrected_lambda_output), index = True)
