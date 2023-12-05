import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
import astropy.units as u

import os
import sys

sys.path.append('../src/')

import matplotlib.pyplot as plt

from event_manip import get_integrated_exposure_between
from hist_manip import data_2_binned_errorbar
from hist_manip import data_2_binned_content
from fit_routines import fit_expGauss, perform_fit_gumble
import hist_manip

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#maybe consider merging this function with the previous one
def get_lambda_dist_per_dec(list_of_files):

    bin_contents_list = []
    bin_error_95 = []
    lambda_dist_list = []
    lower_error_band = []
    upper_error_band = []

    #define the bins in declination
    dec_bin_edges = np.linspace(-90, 90, 91)
    dec_bin_centers = hist_manip.get_bin_centers(dec_bin_edges)
    dec_bin_width = hist_manip.get_bin_width(dec_bin_edges)

    #loop over files
    for i, file in enumerate(list_of_files):

        data = pd.read_parquet(file, engine='fastparquet')

        dec_data = data['dec_center'].to_numpy()
        lambda_values = data['lambda'].to_numpy()

        lambda_values = [lambda_values[np.where( (dec_data > dec_bin_edges[j -1 ]) & (dec_data < dec_bin_edges[j]))[0]] for j in range(1, len(dec_bin_edges))]

        #lambda_values = np.array(lambda_values)

        lambda_dist_list.append(lambda_values)

    #transform list into array
    lambda_values_array = np.array(lambda_dist_list, dtype=list)

    #transpose
    lambda_dist_per_dec_bin = np.transpose(lambda_values_array)

    #print(lambda_dist_per_dec_bin.shape)
    total_lambda_dist_per_dec_bin = np.array([np.concatenate(lambda_dist).ravel() for lambda_dist in lambda_dist_per_dec_bin], dtype=object)

    #print(total_lambda_dist_per_dec_bin[0])

    #create limits for lambda dist
    lambda_dist_edges = np.linspace(-10, 70, 200)
    lambda_bin_centers = hist_manip.get_bin_centers(lambda_dist_edges)

    #compute the .9 quantile of the Lambda distribution
    quantile_99 = [np.nanquantile(lambda_dist, .99) if len(lambda_dist) > 0 else np.nan for lambda_dist in total_lambda_dist_per_dec_bin]
    lambda_bin_content_array = np.array([data_2_binned_content(lambda_dist, lambda_dist_edges, lambda_dist_edges[0], lambda_dist_edges[-1], np.ones(len(lambda_dist)), False) for lambda_dist in total_lambda_dist_per_dec_bin ])
    lambda_bin_centers = np.array([lambda_bin_centers for i in range(len(dec_bin_centers))])

    #build dataframe with lambda_dist
    lambda_dist_df = pd.DataFrame(zip(dec_bin_edges[:-1], dec_bin_edges[1:], lambda_bin_centers, lambda_bin_content_array, quantile_99), columns=['dec_low_edges', 'dec_upper_edges', 'lambda_bin_centers', 'lambda_bin_content', 'lambda_dist_quantile_99'])

    return lambda_dist_df

#maybe consider merging this function with the previous one
def get_lambda_dist_per_rate(filelist):

    #since all the files have the same binning, compute the min and max expected number of events
    mu_per_target = np.unique(pd.read_parquet(filelist[0], engine = 'fastparquet')['mu_in_target'])
    mu_min = np.floor(np.min(mu_per_target)).astype('int')
    mu_max = np.ceil(np.max(mu_per_target)).astype('int')

    #define the binning in the number of expected events
    mu_bins = np.append(np.arange(mu_min, mu_max, .5), mu_max)

    #define the binning in lambda
    lambda_bins = np.append(np.arange(-15, 170, 1), 170)
    lambda_corrected_bins = np.append(np.arange(-30, 120, 1), 120)

    #save the contents of the lambda distributions
    lambda_content_per_mu_list = []
    corrected_lambda_content_per_mu_list = []

    for file in filelist:

        #save relevant columns of dataframe
        data = pd.read_parquet(file, engine = 'fastparquet')

        #save the relevant columns from the dataframe
        mu_per_target = data['mu_in_target'].to_numpy()
        lambda_per_target = data['lambda'].to_numpy()
        lambda_corrected_per_target = data['lambda_corrected'].to_numpy()

        #sort arrays according to mu
        sorted_indices = mu_per_target.argsort()
        mu_per_target, lambda_per_target, lambda_corrected_per_target = mu_per_target[sorted_indices], lambda_per_target[sorted_indices], lambda_corrected_per_target[sorted_indices]

        #split arrays of estimators according to expected number of events in target
        split_indices = np.searchsorted(mu_per_target, mu_bins)[1:-1]
        lambda_per_mu = np.split(lambda_per_target, split_indices)
        corrected_lambda_per_mu = np.split(lambda_corrected_per_target, split_indices)

        #compute the bin contents of the lambda distribution
        lambda_bin_content_per_mu = np.array([np.histogram(array, bins = lambda_bins)[0] for array in lambda_per_mu])
        lambda_corrected_bin_content_per_mu = np.array([np.histogram(array, bins = lambda_corrected_bins)[0] for array in corrected_lambda_per_mu])

        #save lambda bin contents
        lambda_content_per_mu_list.append(lambda_bin_content_per_mu)
        corrected_lambda_content_per_mu_list.append(lambda_corrected_bin_content_per_mu)

    #compute average lambda distribution
    lambda_content_per_mu_list = np.array(lambda_content_per_mu_list)
    corrected_lambda_content_per_mu_list = np.array(corrected_lambda_content_per_mu_list)

    lambda_content_per_mu = np.mean(lambda_content_per_mu_list, axis = 0)
    corrected_lambda_content_per_mu = np.mean(corrected_lambda_content_per_mu_list, axis = 0)

    #build dataframes with lambda distributions
    lambda_dist_data = pd.DataFrame(zip(mu_bins[:-1], mu_bins[1:], np.tile(lambda_bins, (len(mu_bins[1:]), 1)), lambda_content_per_mu), columns=['mu_low_edges', 'mu_upper_edges', 'lambda_bin_edges', 'lambda_bin_content'])

    plt.plot(lambda_corrected_bins[1:], corrected_lambda_content_per_mu[-2])
    plt.plot(lambda_bins[1:], lambda_content_per_mu[-2])
    plt.yscale('log')
    plt.show()

    print(lambda_dist_data)

    #return lambda_dist_df

#define the main function
if __name__ == '__main__':

    #define important quantities
    dec_center = np.radians(-30)
    ra_center = np.radians(0)
    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    input_path = './datasets/iso_estimators/decCenter_%.0f' % np.degrees(dec_center)

    filelist = []

    # loop over files in the directory
    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and 'Estimators' in filename: # and 'Scrambled' not in f:

            filelist.append(filename)


    #lambda_dist_per_dec = get_lambda_dist_per_dec(file_list)
    get_lambda_dist_per_rate(filelist)

    #print(lambda_dist_per_dec.head(20))
    #print(lambda_dist_per_rate.head(20))

    #lambda_dist_per_dec.to_json('./datasets/estimator_dist/Lambda_dist_per_dec_%i.json' % len(file_list), index = True)
    #lambda_dist_per_rate.to_json('./datasets/estimator_dist/Lambda_dist_per_rate_%i.json' % len(file_list), index = True)

# #save the pdf of the directional exposure for each bin in sin(dec)
# low_lims = [-90, -40, 0]
# high_lims = [-80, -30, 10]
#
# for i in range(len(low_lims)-2):
#
#     #bin_centers_tau, bin_content_tau, lower_band_tau, upper_band_tau = get_estimator_dist(file_list, 'tau')
#     bin_centers, bin_content, lower_band, upper_band = get_estimator_dist(file_list, 'lambda', low_lims[i], high_lims[i])
#
#     #expected number of events the requested declination band
#     exp_n_event = 1e5*get_integrated_exposure_between(np.radians(low_lims[i]), np.radians(high_lims[i]), 64, np.radians(80), lat_pao)
#
#     print(exp_n_event)
#     params_init = [.1*max(bin_content), -.5, .2]
#     lower_bounds = [0, -1, 0]
#     upper_bounds = [10*max(bin_content), 1, 1]
#
#     params_opt, params_error, lambda_cont, pdf_cont, chi2 = perform_fit_gumble(bin_centers, bin_content, upper_band - bin_centers, params_init, lower_bounds, upper_bounds)
#
#     print(params_opt)
#     print(chi2)
#
#     plt.plot(bin_centers, bin_content)
#     plt.plot(lambda_cont, pdf_cont)
#
#     plt.fill_between(bin_centers, lower_band, upper_band, alpha = .5)
#
# #bin_centers_nMax, bin_content_nMax, lower_band_nMax, upper_band_nMax = get_estimator_dist(file_list, 'nMax_1day')
# plt.yscale('log')
# plt.ylim(1e-3,1e3)
# plt.show()
