import numpy as np
import pandas as pd
import healpy as hp
import math

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

from scipy.interpolate import Akima1DInterpolator as akima_spline
#from scipy.interpolate import interp1d
from scipy.stats import poisson
from scipy.stats import gaussian_kde

import sys
import os

sys.path.append('../src/')

import matplotlib.pyplot as plt

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map

from hist_manip import get_bin_centers

#compute the pvalue for fixed bin in mu
def get_lambda_pvalue(index, lambda_dist, lambda_array):

    #get the needed parameters
    cdf_bin_centers = get_bin_centers(np.array(lambda_dist.at[index, 'lambda_bin_edges']))
    cdf_bin_content = np.array(lambda_dist.at[index, 'cdf_lambda_bin_content'])
    fit_initial = lambda_dist.at[index, 'fit_init']
    tail_slope = lambda_dist.at[index, 'tail_slope']

    cdf_bin_content = cdf_bin_content / cdf_bin_content[-1]

    #initialize p_value array
    lambda_pvalues = np.empty(lambda_array.shape)

    #estimate and interpolate the cdf
    is_discrete_cdf = cdf_bin_centers <= fit_initial

    discrete_cdf = cdf_bin_content[is_discrete_cdf]
    interpolated_cdf = akima_spline(cdf_bin_centers[is_discrete_cdf], discrete_cdf)
    #interpolated_cdf = interp1d(cdf_bin_centers[is_discrete_cdf], discrete_cdf, kind = 'linear')

    x = np.linspace(cdf_bin_centers[0], cdf_bin_centers[-1], 10000)

    #if index == 0:
        #print(discrete_cdf)
        #print(interpolated_cdf(x))

    #compute the p-value for the discrete cdf
    lambda_below_fit_init = lambda_array < fit_initial

    lambda_pvalues[lambda_below_fit_init] = 1 - interpolated_cdf(lambda_array[lambda_below_fit_init])

    #if lambda_value is above initial point the p_value is analytical
    lambda_above_fit_init = lambda_array >= fit_initial
    lambda_pvalues[lambda_above_fit_init] = (1 - cdf_bin_content[is_discrete_cdf][-1])*np.exp(-tail_slope*(lambda_array[lambda_above_fit_init] - fit_initial))

    return lambda_pvalues

#computes p value given lambda and declination band
def compute_pvalues(file, lambda_dist_file, corrected_lambda_dist_file):

    #save the corresponding data
    lambda_dist = pd.read_json(lambda_dist_file)
    corrected_lambda_dist = pd.read_json(corrected_lambda_dist_file)

    #save the average value bins
    mu_bins = np.append(lambda_dist['mu_low_edges'].values, np.array(lambda_dist['mu_upper_edges'].values)[-1])

    #
    # for i in range(1, len(mu_bins)):
    #
    #     mu_lower = mu_bins[i - 1]
    #     mu_upper = mu_bins[i]
    #
    #     #save all lambda values for all files
    #     lambda_per_mu = []
    #     corrected_lambda_per_mu = []
    #
    #     for file in filelist:

    data = pd.read_parquet(file, engine='fastparquet')

            #cut data based in the requested mu interval
            #data = data[np.logical_and(data['mu_in_target'] > mu_lower, data['mu_in_target'] < mu_upper)]

            #ensure that data is sorted by mu
            #data.sort_values(by=['mu_in_target'], inplace = True, ignore_index = True)

    #save relevant arrays from data frame
    mu_per_target = data['mu_in_target'].to_numpy()
    n_per_target = data['n_events_in_target'].to_numpy()
    lambda_per_target = data['lambda'].to_numpy()
    corrected_lambda_per_target = data['lambda_corrected'].to_numpy()

        #concatenate all lambda values
        #lambda_per_mu = np.array(lambda_per_mu)
        #corrected_lambda_per_mu = np.array(corrected_lambda_per_mu)
        #corrected_lambda_per_mu.append(data['lambda_corrected'].to_numpy())

        #corrected_lambda_per_mu = np.concatenate(corrected_lambda_per_mu)

        #computes lambda pvalues
        #lambda_pvalues = get_lambda_pvalue(i - 1, lambda_dist, lambda_per_mu)
        #lambda_pvalues = [get_lambda_pvalue(i - 1, lambda_dist, lambda_array) for i, lambda_array in enumerate(lambda_per_mu)])

        #print(lambda_per_mu.shape)

    #compute the poisson pvalue
    poisson_pvalues = 1 - .5*(poisson.cdf(n_per_target - 1, mu_per_target) + poisson.cdf(n_per_target, mu_per_target))

    #split the targets by bins in mu
    split_indices = np.searchsorted(mu_per_target, mu_bins)[1:-1]
    splitted_mu_per_target = np.split(mu_per_target, split_indices)
    splitted_lambda_per_target = np.split(lambda_per_target, split_indices)
    splitted_corrected_lambda_per_target = np.split(corrected_lambda_per_target, split_indices)

    #compute pvalue for the splitted lambda per target
    lambda_pvalues = np.concatenate([get_lambda_pvalue(i, lambda_dist, lambda_array) for i, lambda_array in enumerate(splitted_lambda_per_target)])
    corrected_lambda_pvalues = np.concatenate([get_lambda_pvalue(i, corrected_lambda_dist, lambda_array) for i, lambda_array in enumerate(splitted_corrected_lambda_per_target)])

    #save output in a dataframe
    column_names = ['poisson_pvalues', 'lambda_pvalues', 'corrected_lambda_pvalues']
    output_data = pd.DataFrame(zip(poisson_pvalues, lambda_pvalues, corrected_lambda_pvalues))

    pvalue_data = data.copy()
    pvalue_data[column_names] = output_data

    return pvalue_data

#only execute code if this module is main
if __name__ == '__main__':

    #define important quantities
    dec_center = np.radians(-30)
    ra_center = np.radians(0)
    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    #define the input path
    input_path = './datasets/iso_estimators/decCenter_%.0f' % np.degrees(dec_center)
    file_substring = 'targetRadius_%.1f_IsoDist_decCenter_%.0f_raCenter_%.0f_patchRadius_%.0f' % (np.degrees(target_radius), np.degrees(dec_center), np.degrees(ra_center), np.degrees(patch_radius))

    #initialize list to hold files with lambda for each target
    filelist = []

    # loop over files in the directory
    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and file_substring in filename: # and 'Scrambled' not in f:

            filelist.append(filename)

    #print warning if no files found
    if len(filelist) == 0:
        print('No files found!')
        exit()

    #save file containing distribution of lambda as a function of rate
    lambda_dist_path = './datasets/lambda_dist'

    file_lambda_dist = 'CDF_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_10000.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_kde_lambda_dist = 'CDF_GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_100.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_corrected_lambda_dist = 'CDF_GaussianKernelEstimated_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_100.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_lambda_dist = os.path.join(lambda_dist_path, file_lambda_dist)
    file_kde_lambda_dist = os.path.join(lambda_dist_path, file_kde_lambda_dist)
    file_corrected_lambda_dist = os.path.join(lambda_dist_path, file_corrected_lambda_dist)

    #check if both requested file exist
    if (not os.path.exists(file_lambda_dist)) or (not os.path.exists(file_lambda_dist)):
        print('One of the requested files does not exist!')
        exit()

    start = datetime.now()

    lambda_pvalue_average = []
    kde_lambda_pvalue_average = []

    for i, file in enumerate(filelist):

        pvalue_data = compute_pvalues(file, file_lambda_dist, file_corrected_lambda_dist)
        kde_pvalue_data = compute_pvalues(file, file_kde_lambda_dist, file_corrected_lambda_dist)

        lambda_pvalue_average.append(pvalue_data['lambda_pvalues'].values)
        kde_lambda_pvalue_average.append(kde_pvalue_data['lambda_pvalues'].values)

        if i % 10 == 0:
            print('%i / %i files processed!' % (i, len(filelist)))

    print('Took', datetime.now() - start, 's to compute pvalues for all files')

    lambda_pvalue_average = np.ravel(lambda_pvalue_average)
    kde_lambda_pvalue_average = np.ravel(kde_lambda_pvalue_average)

    #print(lambda_pvalue_average.shape)

    x = np.linspace(-6, 0)
    plt.hist(lambda_pvalue_average, bins = 100, range = [0, 1], histtype = 'step')
    plt.hist(kde_lambda_pvalue_average, bins = 100, range = [0, 1], histtype = 'step')
    #plt.plot(x, (len(lambda_pvalue_average) * 6 / 100)*np.log(10)*np.power(10, x))
    #plt.yscale('log')
    #plt.hist(poisson_pvalue_average, bins = 100, histtype = 'step')
    plt.show()
