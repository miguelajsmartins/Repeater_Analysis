import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
import astropy.units as u

import os
import sys

sys.path.append('./src/')

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
    dec_bin_edges = np.linspace(-90, 90, 37)
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

    lambda_bin_content_array = np.array([data_2_binned_content(lambda_dist, lambda_dist_edges, lambda_dist_edges[0], lambda_dist_edges[-1], np.ones(len(lambda_dist)), False) for lambda_dist in total_lambda_dist_per_dec_bin ])
    lambda_bin_centers = np.array([lambda_bin_centers for i in range(len(dec_bin_centers))])

    #build dataframe with lambda_dist
    lambda_dist_df = pd.DataFrame(zip(dec_bin_edges[:-1], dec_bin_edges[1:], lambda_bin_centers, lambda_bin_content_array), columns=['dec_low_edges', 'dec_upper_edges', 'lambda_bin_centers', 'lambda_bin_content'])

    return lambda_dist_df


#save names of files containing events
path_to_files = './datasets/estimators/'
file_list = []

# Loop over files in the directory
for filename in os.listdir(path_to_files):

    f = os.path.join(path_to_files, filename)

    if os.path.isfile(f) and 'Estimator' in f: # and 'Scrambled' not in f:

        file_list.append(f)

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

lambda_dist = get_lambda_dist_per_dec(file_list)

print(lambda_dist.head(10))

lambda_dist.to_json('./datasets/estimator_dist/Lambda_dist_per_dec_%i.json' % len(file_list), index = True)

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
