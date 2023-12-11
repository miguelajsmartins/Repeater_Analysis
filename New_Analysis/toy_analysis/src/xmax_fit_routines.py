import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib.patches as mpatches

from scipy.stats import moment
from scipy.optimize import curve_fit
from scipy.special import erfc

import os
import sys

#defines the gumble distribution
def gumble_fit(x, norm, mean, scale):

    z = (x - mean) / scale

    return norm*np.exp(-z - np.exp(-z))

#performs the fit and outputs the graph of the fit function
def perform_fit_gumble(bin_centers, bin_content, bin_error):

    #makes sure that some lists are array
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #restrict bin contents to non-zero values
    non_zero_bins = bin_content != 0

    bin_centers = bin_centers[non_zero_bins]
    bin_content = bin_content[non_zero_bins]
    bin_error = bin_error[non_zero_bins]

    #compute reasonable guesses for the parameters
    mean = bin_centers[bin_content.argmax()]
    second_moment = np.sum(bin_centers**2 * bin_content) / np.sum(bin_content)
    scale = (np.sqrt(6) / np.pi)*np.sqrt(second_moment**2 - mean**2)
    norm = bin_content.max()

    params_init = [norm, mean, scale]

    #bounds for parameters
    lower_bounds = [0, mean - scale, 0]
    upper_bounds = [sum(bin_content), mean + scale, 10*scale]

    #perform fit
    popt, pcov = curve_fit(gumble_fit, bin_centers, bin_content, p0 = params_init, bounds=(lower_bounds, upper_bounds), sigma=bin_error)

    #errors of parameters
    perr = np.sqrt(np.diag(pcov))

    #produce arrays to plot the fit function
    x = np.linspace(min(bin_centers), max(bin_centers), 5000)
    y = gumble_fit(x, *popt)

    #compute chi2
    ndf = len(bin_content) - len(popt)
    y_exp = np.array(gumble_fit(bin_centers, *popt))
    chi2 = sum(np.power(y_exp - bin_content, 2) / np.power(bin_error, 2)) / ndf

    return [popt, perr, x, y, chi2]
