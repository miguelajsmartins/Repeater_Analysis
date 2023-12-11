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
from fit_routines import perform_fit_exp
from xmax_fit_routines import perform_fit_gumble

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
def get_lambda_dist_per_mu(index, lambda_dist_per_mu):

    #save correspoding value of mu
    mu_low_edge = lambda_dist_per_mu['mu_low_edges'].loc[index]
    mu_upper_edge = lambda_dist_per_mu['mu_upper_edges'].loc[index]

    #save the bins edges of the lambda distribution
    lambda_bin_edges = np.array(lambda_dist_per_mu['lambda_bin_edges'].loc[index])
    lambda_bin_content = np.array(lambda_dist_per_mu['lambda_bin_content'].loc[index])#this must be edited!!!!

    #compute bin centers
    lambda_bin_centers = get_bin_centers(lambda_bin_edges)

    #compute bin errors
    lambda_bin_error = np.sqrt(lambda_bin_content)

    #normalize distribution
    integral = np.trapz(lambda_bin_content, x = lambda_bin_centers)

    lambda_bin_content = lambda_bin_content / integral
    lambda_bin_error = lambda_bin_error / integral

    return mu_low_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error

#define a function to define the style of a color bar
def create_colorbar(fig, ax, colormap, title, limits, label_size):

    cb = fig.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(vmin=limits[0], vmax=limits[1]), cmap = colormap), ax = ax)

    cb.ax.set_ylabel(title, fontsize=label_size)
    cb.ax.set_ylim(limits[0], limits[1])
    cb.ax.tick_params(labelsize=label_size)

# #get extremely fine
# def get_total_kde(input_path, file_kde_lambda_dist, file_kde_corrected_lambda_dist):
#
#     kde_lambda_dist = []
#     kde_corrected_dist = []
#
#     for file in os.listdir(input_path):
#
#         filename = os.path.join(input_path, file)
#
#         if os.path.isfile(filename) and file_kde_lambda_dist in filename:
#
#             with open(filename, 'rb') as f:
#                 kde_lambda_dist.append(pickle.load(f)[:,-1])
#
#     kde_lambda_dist = lambda x: sum(kde(x) for kde in kde_lambda_dist[:,])
#
#     print(kde_lambda_dist)

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
    file_kde_lambda_dist = 'GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_200.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_kde_corrected_lambda_dist = 'GaussianKernelEstimated_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_200.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_kde_lambda_dist = os.path.join(input_path, file_kde_lambda_dist)
    file_kde_corrected_lambda_dist = os.path.join(input_path, file_kde_corrected_lambda_dist)

    #save the corresponding dataframes
    lambda_dist_per_mu = pd.read_json(file_lambda_dist)
    corrected_lambda_dist_per_mu = pd.read_json(file_corrected_lambda_dist)

    # #set position of the pierre auger observatory
    # lat_pao = np.radians(-35.15) # this is the average latitude
    # long_pao = np.radians(-69.2) # this is the averaga longitude
    # height_pao = 1425*u.meter # this is the average altitude
    #
    # #define the earth location corresponding to pierre auger observatory
    # pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)
    #
    # #define theta_max
    # theta_max = np.radians(80)

    # ------------------------------------------
    # Plot the distribution of Lambda and corrected Lambda for each mu value. Fits the distribution as well
    # ------------------------------------------
    quantile = 0.95

    #initialize figure
    fig_lambda_dist = plt.figure(figsize = (15, 5))

    fig_lambda_dist.suptitle(r'$n_{\mathrm{samples}} = 10^4$, $\psi_{\mathrm{target}} = %.1f^\circ$' % np.degrees(target_radius), fontsize = 14)

    ax_lambda_dist = plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=2, fig = fig_lambda_dist)
    ax_corrected_lambda_dist = plt.subplot2grid((3, 3), (0, 1), colspan=1, rowspan=2, fig = fig_lambda_dist)
    ax_kde_lambda_dist = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_kde_corrected_lambda_dist = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_beta_mu_func = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1, fig = fig_lambda_dist)
    ax_mean_mu_func = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2, fig = fig_lambda_dist)

    #get the colormap
    colormap = plt.get_cmap('magma')

    #produce the color array
    mu_array = np.array(lambda_dist_per_mu['mu_low_edges'].values)
    color_array = (lambda_dist_per_mu['mu_low_edges'].values - lambda_dist_per_mu['mu_low_edges'].min()) / (lambda_dist_per_mu['mu_low_edges'].max() - lambda_dist_per_mu['mu_low_edges'].min())
    color_array = colormap(color_array)

    #save moments of distributions
    mean_lambda = []
    sigma_lambda = []
    mean_corrected_lambda = []
    sigma_corrected_lambda = []

    beta_lambda = []
    beta_lambda_error = []
    beta_corrected_lambda = []
    beta_corrected_lambda_error = []

    #save the initial point of the fit
    fit_init_lambda = []
    fit_init_corrected_lambda = []

    #plot the distribution of lambda per expected value of events
    for i in range(len(lambda_dist_per_mu)):

        mu_low_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error = get_lambda_dist_per_mu(i, lambda_dist_per_mu)

        #define the initial point of the fit and fit the lambda distribution
        cdf_lambda = np.cumsum(lambda_bin_content)
        fit_initial =  lambda_bin_centers[cdf_lambda > quantile][0]

        fitted_lambda_dist = perform_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, fit_initial)

        #fill arrays
        mean = np.sum(lambda_bin_centers*lambda_bin_content) / np.sum(lambda_bin_content)
        second_moment = np.sum((lambda_bin_centers**2)*lambda_bin_content) / np.sum(lambda_bin_content)
        sigma = np.sqrt(second_moment - mean**2)

        mean_lambda.append(mean)
        sigma_lambda.append(sigma)
        beta_lambda.append(fitted_lambda_dist[0][1])
        beta_lambda_error.append(fitted_lambda_dist[1][1])
        fit_init_lambda.append(fitted_lambda_dist[2][0])

        print(fitted_lambda_dist[2][0])

        #save the fitted cdf of the lambda distribution
        #cdf_bin_centers, cdf_bin_content, fit_init, tail_slope = get_lambda_cdf([lambda_bin_centers, lambda_bin_content], fitted_lambda_dist)

        #cdf_lambda_dist.append([mu_low_edge, mu_upper_edge, cdf_bin_centers, cdf_bin_content, fit_init, tail_slope])

        #plot distribution for some cases
        if mu_low_edge in [20, 22, 24, 28, 30]:

            #save the kernel density estimation of the lambda pdf
            #kde_lambda_pdf = kde_lambda_dist[i,-1]
            #lambda_cont = np.linspace(lambda_bin_centers[0], lambda_bin_centers[-1], 1000)

            ax_lambda_dist.errorbar(lambda_bin_centers[::2], lambda_bin_content[::2], yerr = lambda_bin_error[::2], color = color_array[i], marker = 'o', linestyle = 'None', markersize = 4, mfc = 'white')
            ax_lambda_dist.plot(fitted_lambda_dist[2], fitted_lambda_dist[3], color = color_array[i])
            #ax_lambda_dist.plot(lambda_cont, kde_lambda_pdf(lambda_cont), color = color_array[i], linestyle = 'dashed')

            #is_positive = lambda_bin_content > 0
            #residuals = (lambda_bin_content[is_positive] - kde_lambda_pdf(lambda_bin_centers[is_positive])) / lambda_bin_content[is_positive]

            #ax_kde_lambda_dist.plot(lambda_bin_centers[is_positive], 100*residuals, color = color_array[i], linestyle = 'dashed')

    #plot the distribution of lambda per expected value of events
    for i in range(len(corrected_lambda_dist_per_mu)):

        mu_low_edge, mu_upper_edge, lambda_bin_centers, lambda_bin_content, lambda_bin_error = get_lambda_dist_per_mu(i, corrected_lambda_dist_per_mu)

        #lambda_bin_content = lambda_bin_content[1:-1] #this must be removed
        #define the initial point of the fit and fit the lambda distribution
        cdf_lambda = np.cumsum(lambda_bin_content)
        fit_initial =  lambda_bin_centers[cdf_lambda > quantile][0]

        fitted_lambda_dist = perform_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, fit_initial)

        #fill arrays
        mean = np.sum(lambda_bin_centers*lambda_bin_content) / np.sum(lambda_bin_content)
        second_moment = np.sum((lambda_bin_centers**2)*lambda_bin_content) / np.sum(lambda_bin_content)
        sigma = np.sqrt(second_moment - mean**2)

        mean_corrected_lambda.append(mean)
        sigma_corrected_lambda.append(sigma)
        beta_corrected_lambda.append(fitted_lambda_dist[0][1])
        beta_corrected_lambda_error.append(fitted_lambda_dist[1][1])
        fit_init_corrected_lambda.append(fitted_lambda_dist[2][0])

        print(fitted_lambda_dist[2][0])

        #save the fitted cdf of the lambda distribution
        #cdf_bin_centers, cdf_bin_content, fit_init, tail_slope = get_lambda_cdf([lambda_bin_centers, lambda_bin_content], fitted_lambda_dist)

        #cdf_corrected_lambda_dist.append([mu_low_edge, mu_upper_edge, cdf_bin_centers, cdf_bin_content, fit_init, tail_slope])

        #plot distribution for some cases
        if mu_low_edge in [20, 22, 24, 28, 30]:

            #save the kernel density estimation of the lambda pdf
            #kde_lambda_pdf = kde_corrected_lambda_dist[i,-1]
            #lambda_cont = np.linspace(lambda_bin_centers[0], lambda_bin_centers[-1], 1000)

            ax_corrected_lambda_dist.errorbar(lambda_bin_centers[::2], lambda_bin_content[::2], yerr = lambda_bin_error[::2], color = color_array[i], marker = 'o', linestyle = 'None', markersize = 3)
            ax_corrected_lambda_dist.plot(fitted_lambda_dist[2], fitted_lambda_dist[3], color = color_array[i])
            #ax_corrected_lambda_dist.plot(lambda_cont, kde_lambda_pdf(lambda_cont), color = color_array[i], linestyle = 'dashed')

            #is_positive = lambda_bin_content > 0
            #residuals = (lambda_bin_content[is_positive] - kde_lambda_pdf(lambda_bin_centers[is_positive])) / lambda_bin_content[is_positive]

            #ax_kde_corrected_lambda_dist.plot(lambda_bin_centers[is_positive], 100*residuals, color = color_array[i], linestyle = 'dashed')

    #plot moments as a function of mean number of events
    ax_mean_mu_func.scatter(sigma_lambda, mean_lambda, edgecolors = color_array, marker = 'o', linestyle = 'None', s = 10, facecolor = 'lightgrey')
    ax_mean_mu_func.scatter(sigma_corrected_lambda, mean_corrected_lambda, c = color_array, marker = 'o', linestyle = 'None', s = 7)

    ax_beta_mu_func.errorbar(mu_array, beta_lambda, yerr = beta_lambda_error, color = 'tab:blue', marker = 'o', linestyle = 'None', markersize = 4, mfc = 'white')
    ax_beta_mu_func.errorbar(mu_array, beta_corrected_lambda, yerr = beta_corrected_lambda_error, color = 'tab:blue', marker = 'o', linestyle = 'None', markersize = 3)

    #define the style of the plot
    ax_lambda_dist.set_yscale('log')
    ax_lambda_dist.set_xlim(-30, 100)
    ax_lambda_dist.set_ylim(1e-7, 1)

    ax_kde_lambda_dist.set_xlim(-30, 100)
    ax_kde_lambda_dist.set_ylim(-20, 20)

    ax_corrected_lambda_dist.set_yscale('log')
    ax_corrected_lambda_dist.set_xlim(-30, 100)
    ax_corrected_lambda_dist.set_ylim(1e-7, 1)

    ax_kde_corrected_lambda_dist.set_xlim(-30, 100)
    ax_kde_corrected_lambda_dist.set_ylim(-20, 20)

    ax_mean_mu_func.set_ylim(-1, 20)
    ax_beta_mu_func.set_ylim(0.15, 0.35)

    #plot colorbars
    create_colorbar(fig_lambda_dist, ax_lambda_dist, colormap, r'$\mu$', [mu_array.min(), mu_array.max()], 14)
    create_colorbar(fig_lambda_dist, ax_corrected_lambda_dist, colormap, r'$\mu$', [mu_array.min(), mu_array.max()], 14)
    create_colorbar(fig_lambda_dist, ax_mean_mu_func, colormap, r'$\mu$', [mu_array.min(), mu_array.max()], 14)

    #define the style of the axis
    ax_lambda_dist = set_style(ax_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)
    ax_corrected_lambda_dist = set_style(ax_corrected_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)
    ax_kde_lambda_dist = set_style(ax_kde_lambda_dist, '', r'$\Lambda$', r'$1 - \frac{\hat{f}_{\Lambda}(\Lambda)}{f_{\Lambda}(\Lambda)} (\%)$', 14)
    ax_kde_corrected_lambda_dist = set_style(ax_kde_corrected_lambda_dist, '', r'$\Lambda$', r'$1 - \frac{\hat{f}_{\Lambda}(\Lambda)}{f_{\Lambda}(\Lambda)} (\%)$', 14)
    ax_mean_mu_func = set_style(ax_mean_mu_func, '', r'$\sigma(\Lambda)$', r'$\langle \Lambda \rangle$', 14)
    ax_beta_mu_func = set_style(ax_beta_mu_func, '', r'$\mu$', r'$\beta$', 14)

    ax_mean_mu_func.set_facecolor('lightgray')
    #save figure
    fig_lambda_dist.tight_layout()
    fig_lambda_dist.savefig(os.path.join(output_path, 'Lambda_distribution_patchRadius_%.0f_targetRadius_%.1f.pdf' % (np.degrees(patch_radius), np.degrees(target_radius))))

    #save the cdfs in the corresponding files
    save_lambda_cdf(input_path, file_kde_lambda_dist, fit_init_lambda, beta_lambda)
    save_lambda_cdf(input_path, file_kde_corrected_lambda_dist, fit_init_corrected_lambda, beta_corrected_lambda)

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
