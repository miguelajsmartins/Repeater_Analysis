import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.special import exp1 as exp_int

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

#compute the values of the weights given a uniform distribution
def get_weights(min, max, size, n_samples):

    #generate uniformly distributed values between min and max
    u = np.random.random((n_samples, size))

    x = (max - min)*u + min

    #compute the expected rate
    exp_rate = size / (max - min)

    #order x and compute difference between consecutive events
    ordered_x = np.sort(x, axis = 1)
    diff_x = np.diff(ordered_x)*exp_rate

    #compute the weights
    weights = - np.log(diff_x)

    return weights

#compute the theoretical expectation for the average of the weights
def compute_teo_average(x):

    norm = 1 / (1 - np.exp(-x))

    return norm*(np.euler_gamma + exp_int(x) + np.log(x)*np.exp(-x))

#compute the theoretical pdf followed by the weights
def compute_teo_pdf(x, size):

    norm = 1 / (1 - np.exp(-size))

    min = -np.log(size)

    #mask array
    result = np.ones(len(x))
    above_min = x > min

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
    size_array = np.linspace(2, 30, 29, dtype = 'int')
    n_samples = int(1e5)

    #save colormap
    colormap = plt.get_cmap('coolwarm')

    #initialize figure with distribution of weights, first and second moments evolution
    fig_weights_dist = plt.figure(figsize=(15, 4))

    ax_weights_dist = fig_weights_dist.add_subplot(1, 3, 1)
    ax_weights_average = fig_weights_dist.add_subplot(1, 3, 2)
    ax_weights_width = fig_weights_dist.add_subplot(1, 3, 3)

    #initialize arrays to hold the moments
    average = []
    width = []

    for i, size in enumerate(size_array):

        #get all the weights regardless of the sample index
        weights = get_weights(rand_min, rand_max, size, n_samples)[:,0]

        #save the moments of the distributions
        average.append(weights.mean())
        width.append(weights.std())

        #build distribution
        weights_bin_centers, weights_bin_content, weights_bin_error = data_2_binned_errorbar(weights, 200, -4, 20, np.ones(len(weights)), False)

        print('Mean = %i, Mode = %.2f' % (size, weights_bin_centers[weights_bin_content.argmax()]))

        #build the distribution of weights for some values of expected number of events
        if i == 25:

            #normalize the distribution
            integral = np.trapz(weights_bin_content, x = weights_bin_centers)
            weights_bin_content = weights_bin_content / integral
            weights_bin_error = weights_bin_error / integral

            #compute the teoretical pdf
            weights_cont = np.linspace(-5, 20, 1000)
            teo_pdf = compute_teo_pdf(weights_cont, size)

            #compute a chi2
            non_zero = weights_bin_content > 0
            #first_above_zero = weights_bin_centers[above_zero][0]

            chi2 = np.sum((compute_teo_pdf(np.array(weights_bin_centers[non_zero]), size) - weights_bin_content[non_zero])**2 / weights_bin_content[non_zero])

            print('Chi2 = %.2f' % chi2)

            ax_weights_dist.errorbar(weights_bin_centers, weights_bin_content, yerr = weights_bin_error, marker = 'o', markersize = 3, linestyle = 'None')
            ax_weights_dist.plot(weights_cont, teo_pdf, linestyle = 'solid')

    #compute the teoretical moments
    mean_cont = np.linspace(size_array[0], size_array[-1], 1000)
    average_prediction = compute_teo_average(mean_cont)

    #plot the moments
    ax_weights_average.plot(size_array, average, color = 'tab:blue', linestyle = 'None', marker = 'o', markersize = 4)
    ax_weights_average.plot(mean_cont, average_prediction, linestyle = 'dashed', color = 'tab:orange')

    ax_weights_width.plot(size_array, width, linestyle = 'None', marker = 'o', markersize = 4)

    #define the style of the axis
    ax_weights_dist.set_yscale('log')
    ax_weights_dist.set_ylim(1e-6, 5e-1)

    ax_weights_dist = set_style(ax_weights_dist, '', r'$w$', r'$f_{W}(w)$', 14)
    ax_weights_average = set_style(ax_weights_average, '', r'$\mu$', r'$\langle w \rangle$', 14)
    ax_weights_width = set_style(ax_weights_width, '', r'$\mu$', r'$\sigma(w)$', 14)

    #save figure
    fig_weights_dist.tight_layout()
    fig_weights_dist.savefig(os.path.join(output_path, 'weights_distribution.pdf'))
