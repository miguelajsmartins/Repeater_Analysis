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
from scipy.stats import poisson
from scipy.stats import gaussian_kde

import pickle

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

#computes p value given lambda and declination band
def get_gaussian_kernel(filelist, lambda_dist_file, corrected_lambda_dist_file):

    #save the corresponding data
    lambda_dist = pd.read_json(lambda_dist_file)
    corrected_lambda_dist = pd.read_json(corrected_lambda_dist_file)

    #save the average value bins
    mu_bins = np.append(lambda_dist['mu_low_edges'].values, np.array(lambda_dist['mu_upper_edges'].values)[-1])

    #save the kernel density estimations for the lambda pdf
    kernel_lambda_pdf_per_mu = []
    kernel_corrected_lambda_pdf_per_mu = []

    for i in range(1, len(mu_bins)):

        mu_lower = mu_bins[i - 1]
        mu_upper = mu_bins[i]

        #save all lambda values for all files
        lambda_per_mu = []
        corrected_lambda_per_mu = []

        for file in filelist:

            data = pd.read_parquet(file, engine='fastparquet')

            #cut data based in the requested mu interval
            data = data[np.logical_and(data['mu_in_target'] > mu_lower, data['mu_in_target'] < mu_upper)]

            #save the lambda values
            lambda_per_mu.append(data['lambda'].to_numpy())
            corrected_lambda_per_mu.append(data['lambda_corrected'].to_numpy())

        #concatenate all lambda values
        lambda_per_mu = np.array(lambda_per_mu).flatten()
        corrected_lambda_per_mu = np.array(corrected_lambda_per_mu).flatten()

        #estimate the lamda pdf using a bootstrap gaussian kernel density estimaton
        kernel_lambda_pdf = gaussian_kde(lambda_per_mu)
        kernel_corrected_lambda_pdf = gaussian_kde(corrected_lambda_per_mu)

        kernel_lambda_pdf_per_mu.append(kernel_lambda_pdf)
        kernel_corrected_lambda_pdf_per_mu.append(kernel_corrected_lambda_pdf)

        print('%i / %i bins in mu done!' % (i, len(mu_bins)-1))

    #tranform lists into array for easier manipulation
    kernel_lambda_pdf_per_mu = np.array(list(zip(mu_bins[:-1], mu_bins[1:], kernel_lambda_pdf_per_mu)))
    kernel_corrected_lambda_pdf_per_mu = np.array(list(zip(mu_bins[:-1], mu_bins[1:], kernel_corrected_lambda_pdf_per_mu)))

    return kernel_lambda_pdf_per_mu, kernel_corrected_lambda_pdf_per_mu

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
    file_corrected_lambda_dist = 'CDF_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_10000.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_lambda_dist = os.path.join(lambda_dist_path, file_lambda_dist)
    file_corrected_lambda_dist = os.path.join(lambda_dist_path, file_corrected_lambda_dist)

    #check if both requested file exist
    if (not os.path.exists(file_lambda_dist)) or (not os.path.exists(file_lambda_dist)):
        print('One of the requested files does not exist!')
        exit()

    #estimate the lambda pdf using gaussian kernel density
    start = datetime.now()

    kernel_lambda_pdf_per_mu, kernel_corrected_lambda_pdf_per_mu = get_gaussian_kernel(filelist, file_lambda_dist, file_corrected_lambda_dist)

    print('Took', datetime.now() - start, 's to compute pvalues for all files')

    #save the kernel density estimations in pickled files
    output_lambda = 'GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.pkl' % (np.degrees(patch_radius), np.degrees(target_radius))
    output_corrected_lambda = 'GaussianKernelEstimated_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.pkl' % (np.degrees(patch_radius), np.degrees(target_radius))

    with open(os.path.join(lambda_dist_path, output_lambda), 'wb') as file:
        pickle.dump(kernel_lambda_pdf_per_mu, file)

    with open(os.path.join(lambda_dist_path, output_corrected_lambda), 'wb') as file:
        pickle.dump(kernel_corrected_lambda_pdf_per_mu, file)
