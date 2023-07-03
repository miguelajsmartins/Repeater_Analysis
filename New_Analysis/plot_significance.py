import pandas as pd
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

sys.path.append('./src/')

import hist_manip
from hist_manip import data_2_binned_errorbar
from event_manip import compute_directional_exposure
from axis_style import set_style

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#maybe consider merging this function with the previous one
# def get_Lambda_p_value_dist(list_of_files):
#
#     #initialize the li ma significance list
#     LiMa_significance_bin_content = []
#     lower_error_band = []
#     upper_error_band = []
#
#     #loop over files
#     for i, file in enumerate(list_of_files):
#
#         data = pd.read_parquet(file, engine='fastparquet')
#
#         if 'lambda_p_value' not in data.columns:
#             continue
#
#         LiMa_significance = data['lambda_p_value'].to_numpy()
#
#         plt.hist(LiMa_significance, bins = 100, range = [0, 1])
#         plt.show()
#
#         bin_centers, bin_contents, bin_error = data_2_binned_errorbar(LiMa_significance, 100, 0, 1, np.ones(len(LiMa_significance)), False)
#
#         if i == 0:
#             LiMa_significance_bin_centers = bin_centers
#
#         LiMa_significance_bin_content.append(bin_contents)
#
#     #compute average Li Ma significance
#     average_bin_content = np.mean(LiMa_significance_bin_content, axis = 0)
#
#     #compute bands corresponding to 1 sigma
#     fluctuations = np.std(LiMa_significance_bin_content, axis = 0)
#
#     lower_error_band = average_bin_content - 2*fluctuations
#     upper_error_band = average_bin_content + 2*fluctuations
#
#     return LiMa_significance_bin_centers, average_bin_content, lower_error_band, upper_error_band

#maybe consider merging this function with the previous one
def get_LiMa_significance_dist(list_of_files):

    #initialize the li ma significance list
    LiMa_significance_bin_content = []
    lower_error_band = []
    upper_error_band = []

    #loop over files
    for i, file in enumerate(list_of_files):

        data = pd.read_parquet(file, engine='fastparquet')

        #restrict regions with appreceable exposure
        #data = data[data['dec_center'] < 0]

        LiMa_significance = data['LiMa_significance'].to_numpy()

        bin_centers, bin_contents, bin_error = data_2_binned_errorbar(LiMa_significance, 100, -6, 6, np.ones(len(LiMa_significance)), False)

        if i == 0:
            LiMa_significance_bin_centers = bin_centers

        LiMa_significance_bin_content.append(bin_contents)

    #compute average Li Ma significance
    average_bin_content = np.mean(LiMa_significance_bin_content, axis = 0)

    #compute bands corresponding to 1 sigma
    fluctuations = np.std(LiMa_significance_bin_content, axis = 0)

    lower_error_band = average_bin_content - 2*fluctuations
    upper_error_band = average_bin_content + 2*fluctuations

    return LiMa_significance_bin_centers, average_bin_content, lower_error_band, upper_error_band

#save names of files containing events
#path_to_files = './datasets/scrambled_events/'
path_to_files_estimator = './datasets/estimators'
estimator_filelist = []

# Loop over files in the directory
for filename in os.listdir(path_to_files_estimator):

    f = os.path.join(path_to_files_estimator, filename)

    if os.path.isfile(f) and 'Scrambled' in f:

        estimator_filelist.append(f)

print(estimator_filelist)

#compute the LiMa significance distribution
LiMa_significance_bin_centers, LiMa_average_bin_content, LiMa_lower_error_band, LiMa_upper_error_band = get_LiMa_significance_dist(estimator_filelist)
#Lambda_pvalue_bin_centers, Lambda_average_bin_content, Lambda_lower_error_band, Lambda_upper_error_band = get_Lambda_p_value_dist(estimator_filelist)

#defines maximum zenith angle
theta_max = np.radians(80)

#plot the significances
fig_significance = plt.figure(figsize=(10, 4))
ax_LiMa_significance = fig_significance.add_subplot(121)
ax_lambda_pvalue = fig_significance.add_subplot(122)

ax_LiMa_significance.plot(LiMa_significance_bin_centers, LiMa_average_bin_content, color = 'tab:blue', label=r'$10^3$ realizations of iso. sky')
ax_LiMa_significance.fill_between(LiMa_significance_bin_centers, LiMa_lower_error_band, LiMa_upper_error_band, alpha = .5)
ax_LiMa_significance = set_style(ax_LiMa_significance, '', r'$\sigma_{\mathrm{LiMa}}$', r'Arb. units', 12)
ax_LiMa_significance.legend(loc='upper right', fontsize = 12)

#ax_lambda_pvalue.plot(Lambda_pvalue_bin_centers, Lambda_average_bin_content, color = 'tab:blue', label=r'$10^3$ realizations of iso. sky')
#ax_lambda_pvalue.fill_between(Lambda_pvalue_bin_centers, Lambda_lower_error_band, Lambda_upper_error_band, alpha = .5)
#ax_lambda_pvalue = set_style(ax_lambda_pvalue, '', r'$\Lambda$ $p$-value', r'Arb. units', 12)
#ax_lambda_pvalue.legend(loc='upper right', fontsize = 12)

fig_significance.tight_layout()
fig_significance.savefig('./results/LiMa_and_Lambda_significances_th%.0f.pdf' % np.degrees(theta_max))

# --------------------
# Notes: this code is not efficient because it loops more than once over many files. Change this
# ---------------------
#numer of files
# n_files = len(file_list)
#
# #set position of the pierre auger observatory
# pao_lat = np.radians(-35.15) # this is the average latitude
# pao_long = np.radians(-69.2) # this is the averaga longitude
# pao_height = 1425*u.meter # this is the average altitude
#
# #define the earth location corresponding to pierre auger observatory
# pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)
#
# #defines the maximum declination
# theta_max = np.radians(80)

#-----------------
# directional exposure as aa function of declination
#-----------------
#array to plot theortical directional exposure as a function of declination
# dec_cont = np.linspace(-np.pi / 2, np.pi / 2, 5000)
#
# theo_directional_exposure = compute_directional_exposure(dec_cont, theta_max, pao_lat)
# integrated_exposure = np.trapz(theo_directional_exposure*np.cos(dec_cont), x=dec_cont)
# theo_directional_exposure = (theo_directional_exposure) / integrated_exposure
#
# #save the pdf of the directional exposure for each bin in sin(dec)
# bin_centers_omega, bin_content_omega, lower_band_omega, upper_band_omega = data_directional_significance_func(file_list)
#
# #plot directional exposure from data and theory
# fig_significance = plt.figure(figsize=(5, 4))
# ax_significance = fig_significance.add_subplot(111)
#
# ax_significance.plot(bin_centers_omega, bin_content_omega, color = 'tab:blue', label=r'$\langle \omega(\delta) \rangle$ from $10^3$ realizations of iso. sky')
# ax_significance.plot(np.sin(dec_cont), theo_directional_exposure, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
# ax_significance.fill_between(bin_centers_omega, lower_band_omega, upper_band_omega, alpha = .5)
# ax_significance = set_style(ax_significance, '', r'$\sin (\delta)$', r'$\omega(\delta)$', 12)
# ax_significance.legend(loc='upper right', fontsize = 12)
#
# fig_significance.tight_layout()
# fig_significance.savefig('./results/directional_significance_func_Theory_%i_IsotropicSkies_th%.0f.pdf' % (n_files, np.degrees(theta_max)))

#------------------------------
# Plot expected number and rate of events
#------------------------------
# NSIDE = 64
# obs_time = 10 #in years, maybe this should be more flexible
# n_events = 1e5 #this should be more flexible
# area_of_sky_bin = 4*np.pi / hp.nside2npix(NSIDE)
#
# #compute theoretical number of events per bin
# theo_event_number_per_bin = (.5 / np.pi)*theo_directional_exposure*n_events*area_of_sky_bin
# theo_event_rate_per_bin = theo_event_number_per_bin / obs_time #in units of per year
#
# #number of events per pixel
# bin_centers_event_density, bin_content_event_density, lower_band_event_density, upper_band_event_density = number_of_events_dec_func(estimator_filelist)
#
# #plot directional exposure from data and theory
# fig_event_number_dec = plt.figure(figsize=(10, 4))
# ax_event_number_dec = fig_event_number_dec.add_subplot(121)
# ax_event_rate_dec = fig_event_number_dec.add_subplot(122)
#
# #for event number
# ax_event_number_dec.plot(bin_centers_event_density, bin_content_event_density, color = 'tab:blue', label=r'Average from $10^3$ realizations of iso. sky')
# ax_event_number_dec.plot(np.sin(dec_cont), theo_event_number_per_bin, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
# ax_event_number_dec.fill_between(bin_centers_event_density, lower_band_event_density, upper_band_event_density, alpha = .5)
# ax_event_number_dec = set_style(ax_event_number_dec, '', r'$\sin (\delta)$', r'Events per pixel', 12)
# ax_event_number_dec.legend(loc='upper right', title = r'$10^5$ events in $10$ yrs. $A_{\mathrm{pix}} = %.5f$ sr' % area_of_sky_bin, title_fontsize = 12, fontsize = 12)
#
# #for event rate
# ax_event_rate_dec.plot(bin_centers_event_density, bin_content_event_density / obs_time, color = 'tab:blue', label=r'Average from $10^3$ realizations of iso. sky')
# ax_event_rate_dec.plot(np.sin(dec_cont), theo_event_rate_per_bin, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
# ax_event_rate_dec.fill_between(bin_centers_event_density, lower_band_event_density / obs_time, upper_band_event_density / obs_time, alpha = .5)
# ax_event_rate_dec = set_style(ax_event_rate_dec, '', r'$\sin (\delta)$', r'$\Gamma \;(\mathrm{yr}^{-1})$', 12)
# ax_event_rate_dec.legend(loc='upper right', title = r'$10^5$ events in $10$ yrs. $A_{\mathrm{pix}} = %.5f$ sr' % area_of_sky_bin, title_fontsize = 12, fontsize = 12)
#
# fig_event_number_dec.tight_layout()
# fig_event_number_dec.savefig('./results/event_rate_per_skybin_data_theo_%i_IsotropicSkies_th%.0f.pdf' % (n_files, np.degrees(theta_max)))
