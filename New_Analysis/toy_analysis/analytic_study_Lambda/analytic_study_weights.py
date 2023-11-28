import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.special import exp1 as exp_int
from scipy.special import gammaincc as gamma_inc
from scipy.special import gamma

import matplotlib.pyplot as plt

#from scipy.interpolate import Akima1DInterpolator as akima_spline

import sys
import os

sys.path.append('../src/')

from hist_manip import data_2_binned_errorbar
from axis_style import set_style

#from fit_routines import perform_fit_exp

#enable latex rendering and latex like style font
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#compute the theoretical expectation for the average of the weights
def compute_teo_average(x):

    first_term = np.euler_gamma + exp_int(x) + np.log(x)*np.exp(-x)
    second_term = 1 - (x*np.log(x) + 1)*np.exp(-x)

    return first_term + (second_term / x)

#compute the values of the weights given a uniform distribution
def get_weights(min, max, mean, n_samples):

    #generate numbers of events following a poisson
    n_events = np.random.poisson(mean, size = n_samples)

    #generate uniformly distributed values between min and max
    u_list = [np.random.random(n_events_per_sample) for n_events_per_sample in n_events]

    #complete list with np.nan to make sure all entries have the same
    u_matrix = np.array([np.append(u, np.full(n_events.max() - len(u), np.nan)) for u in u_list])

    x = (max - min)*u_matrix + min

    #compute the expected rate
    exp_rate = size / (max - min)

    #order x and compute difference between consecutive events
    ordered_x = np.sort(x, axis = 1)
    diff_x = np.diff(ordered_x)*exp_rate

    #compute the weights
    weights = - np.log(diff_x)
    lambda_var = np.nansum(weights, axis = 1)
    corrected_lambda_var = lambda_var - (n_events - 1)*compute_teo_average(mean)

    #filter out nan values
    weights = np.ravel(weights)
    weights = weights[np.logical_not(np.isnan(weights))]

    print('Computed all weights for mean = ', mean)

    #diff_x = np.ravel(diff_x)
    #diff_x = diff_x[np.logical_not(np.isnan(diff_x))]

    return lambda_var, corrected_lambda_var, weights



#compute the theoretical pdf followed by the weights
def compute_teo_pdf(x, mean):

    min = -np.log(mean)

    #mask array
    result = np.ones(len(x))
    above_min = x > min

    norm = (mean + 1 - np.exp(-x[above_min])) / mean
    result[above_min] = norm*np.exp(-x[above_min] - np.exp(-x[above_min]))
    result[np.logical_not(above_min)] = np.nan

    return result

if __name__ == '__main__':

    #define the output directory
    output_path = './results'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #fix the seed
    seed = 10
    np.random.seed(seed)

    #define the interval where to generate random numbers
    rand_min = 0
    rand_max = 10

    #define the size of each sample and the number of samples
    size_array = np.linspace(3, 20, 18, dtype = 'int')
    n_samples = int(1e5)

    #save colormap
    colormap = plt.get_cmap('coolwarm')
    color_array = colormap(np.linspace(0, 1, len(size_array)))

    #initialize figure with distribution of weights, first and second moments evolution
    fig_weights_dist = plt.figure(figsize=(15, 4))
    fig_lambda_dist = plt.figure(figsize=(15, 4))

    #initialize axis
    ax_weights_dist = fig_weights_dist.add_subplot(1, 3, 1)
    ax_weights_average = fig_weights_dist.add_subplot(1, 3, 2)
    ax_weights_width = fig_weights_dist.add_subplot(1, 3, 3)

    #initialize axis
    ax_lambda_dist = fig_lambda_dist.add_subplot(1, 3, 1)
    ax_lambda_average = fig_lambda_dist.add_subplot(1, 3, 2)
    ax_lambda_width = fig_lambda_dist.add_subplot(1, 3, 3)

    #initialize arrays to hold the moments
    average = []
    width = []

    lambda_average = []
    corr_lambda_average = []

    for i, size in enumerate(size_array):

        #get all the weights regardless of the sample index
        lambda_var, corr_lambda_var, weights = get_weights(rand_min, rand_max, size, n_samples)

        #save the moments of the distributions
        average.append(weights.mean())
        width.append(weights.std())

        lambda_average.append(lambda_var.mean())
        corr_lambda_average.append(corr_lambda_var.mean())

        #build distribution
        weights_bin_centers, weights_bin_content, weights_bin_error = data_2_binned_errorbar(weights, 200, -4, 20, np.ones(len(weights)), False)
        lambda_bin_centers, lambda_bin_content, lambda_bin_error  = data_2_binned_errorbar(lambda_var, 50, -15, 50, np.ones(len(lambda_var)), False)
        corr_lambda_bin_centers, corr_lambda_bin_content, corr_lambda_bin_error  = data_2_binned_errorbar(corr_lambda_var, 50, -15, 50, np.ones(len(corr_lambda_var)), False)

        #build the distribution of weights for some values of expected number of events
        if i % 5 == 0:

            #normalize the distribution
            integral_weights = np.trapz(weights_bin_content, x = weights_bin_centers)
            weights_bin_content = weights_bin_content / integral_weights
            weights_bin_error = weights_bin_error / integral_weights

            integral_lambda = np.trapz(lambda_bin_content, x = lambda_bin_centers)
            lambda_bin_content = lambda_bin_content / integral_lambda
            lambda_bin_error = lambda_bin_error / integral_lambda

            integral_corr_lambda = np.trapz(corr_lambda_bin_content, x = corr_lambda_bin_centers)
            corr_lambda_bin_content = corr_lambda_bin_content / integral_corr_lambda
            corr_lambda_bin_error = corr_lambda_bin_error / integral_corr_lambda

            #compute the teoretical pdf
            weights_cont = np.linspace(-5, 20, 1000)
            teo_pdf = compute_teo_pdf(weights_cont, size)

            #compute a chi2
            non_zero = weights_bin_content > 0
            #first_above_zero = weights_bin_centers[above_zero][0]

            chi2 = np.sum((compute_teo_pdf(np.array(weights_bin_centers[non_zero]), size) - weights_bin_content[non_zero])**2 / weights_bin_content[non_zero])

            print('Chi2 = %.2f' % chi2)

            ax_weights_dist.errorbar(weights_bin_centers, weights_bin_content, yerr = weights_bin_error, marker = 'o', markersize = 3, linestyle = 'None', color = color_array[i])
            ax_weights_dist.plot(weights_cont, teo_pdf, linestyle = 'solid', color = color_array[i])

            ax_lambda_dist.errorbar(corr_lambda_bin_centers, corr_lambda_bin_content, yerr = corr_lambda_bin_error, marker = 'o', markersize = 3, linestyle = 'None', color = color_array[i])
            ax_lambda_dist.errorbar(lambda_bin_centers, lambda_bin_content, yerr = lambda_bin_error, marker = 'o', markersize = 4, mfc = 'None', linestyle = 'None', color = color_array[i])

    #compute the teoretical moments
    mean_cont = np.linspace(size_array[0], size_array[-1], 1000)
    average_prediction = compute_teo_average(mean_cont)

    #plot the moments for the weights
    ax_weights_average.plot(size_array, average, color = 'tab:blue', linestyle = 'None', marker = 'o', markersize = 4)
    ax_weights_average.plot(mean_cont, average_prediction, linestyle = 'dashed', color = 'tab:orange')

    ax_weights_width.plot(size_array, width, linestyle = 'None', marker = 'o', markersize = 4)

    #plot the moments for lambda
    ax_lambda_average.plot(size_array, lambda_average, color = 'tab:blue', linestyle = 'None', marker = 'o', markersize = 4, label = 'No correction')
    ax_lambda_average.plot(size_array, corr_lambda_average, color = 'tab:blue', mfc = 'None', marker = 'o', markersize = 5, linestyle = 'None', label = 'Analytical correction')

    #define the style of the axis
    ax_weights_dist.set_yscale('log')
    ax_weights_dist.set_ylim(1e-6, 5e-1)

    ax_lambda_dist.set_yscale('log')
    ax_lambda_dist.set_ylim(1e-5, 5e-1)

    ax_weights_dist = set_style(ax_weights_dist, '', r'$w$', r'$f_{W}(w)$', 14)
    ax_weights_average = set_style(ax_weights_average, '', r'$\mu$', r'$\langle w \rangle$', 14)
    ax_weights_width = set_style(ax_weights_width, '', r'$\mu$', r'$\sigma(w)$', 14)

    ax_lambda_dist = set_style(ax_lambda_dist, '', r'$\Lambda$', r'$f_{\Lambda}(\Lambda)$', 14)
    ax_lambda_average = set_style(ax_lambda_average, '', r'$\mu$', r'$\langle \Lambda \rangle$', 14)

    #draw legend
    ax_lambda_average.legend(loc = 'upper left', fontsize = 14)

    #save figure
    fig_weights_dist.tight_layout()
    fig_lambda_dist.tight_layout()

    fig_weights_dist.savefig(os.path.join(output_path, 'weights_distribution.pdf'))
    fig_lambda_dist.savefig(os.path.join(output_path, 'Lambda_distribution.pdf'))
