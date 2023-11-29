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

def data_directional_exposure_dec_func(list_of_files):

    bin_contents_list = []
    lower_error_band = []
    upper_error_band = []

    #loop over files
    for i, file in enumerate(list_of_files):

        data = pd.read_parquet(file, engine='fastparquet')

        bin_centers, bin_contents, bin_error = data_2_binned_errorbar(np.sin(np.radians(data['dec'])), 100, -1, 1, np.ones(len(data.index)), True)

        if i == 0:
            bin_centers_fix = bin_centers

        bin_contents_list.append(bin_contents)

    #compute average directional exp as a function of sin(dec)
    average_bin_content = np.mean(bin_contents_list, axis = 0)

    #compute bands corresponding to 1 sigma
    fluctuations = np.std(bin_contents_list, axis = 0)

    lower_error_band = average_bin_content - 2*fluctuations
    upper_error_band = average_bin_content + 2*fluctuations

    return bin_centers_fix, average_bin_content, lower_error_band, upper_error_band

#maybe consider merging this function with the previous one
def number_of_events_dec_func(list_of_files):

    bin_contents_list = []
    lower_error_band = []
    upper_error_band = []

    #define the bins in declination
    sin_dec_bin_edges = np.linspace(-1, 1, 100)
    sin_dec_bin_centers = hist_manip.get_bin_centers(sin_dec_bin_edges)
    sin_dec_bin_width = hist_manip.get_bin_width(sin_dec_bin_edges)

    #loop over files
    for i, file in enumerate(list_of_files):

        data = pd.read_parquet(file, engine='fastparquet')

        sin_dec_data = np.sin(np.radians(data['dec_center'].to_numpy()))

        repetitions_in_sin_dec_bin = [len(sin_dec_data[np.where( (sin_dec_data > sin_dec_bin_edges[j -1 ]) & (sin_dec_data < sin_dec_bin_edges[j]))[0] ]) for j in range(1, len(sin_dec_bin_edges))]

        #print(repetitions_in_sin_dec_bin)

        bin_centers, bin_contents, bin_error = data_2_binned_errorbar(np.sin(np.radians(data['dec_center'])), sin_dec_bin_edges, -1, 1, data['events_in_target'], False)

        #print(bin_ce)
        bin_contents = bin_contents / repetitions_in_sin_dec_bin
        bin_error = bin_error / repetitions_in_sin_dec_bin

        if i == 0:
            bin_centers_fix = bin_centers

        bin_contents_list.append(bin_contents)

    #compute average directional exp as a function of sin(dec)
    average_bin_content = np.mean(bin_contents_list, axis = 0)

    #compute bands corresponding to 1 sigma
    fluctuations = np.std(bin_contents_list, axis = 0)

    lower_error_band = average_bin_content - 2*fluctuations
    upper_error_band = average_bin_content + 2*fluctuations

    return bin_centers_fix, average_bin_content, lower_error_band, upper_error_band

#save names of files containing events
path_to_files = './datasets/scrambled_events/'
path_to_files_estimator = './datasets/estimators'
file_list = []
estimator_filelist = []

# Loop over files in the directory
for filename in os.listdir(path_to_files):

    f = os.path.join(path_to_files, filename)

    if os.path.isfile(f): # and 'Scrambled' not in f:

        file_list.append(f)

# Loop over files in the directory
for filename in os.listdir(path_to_files_estimator):

    f = os.path.join(path_to_files_estimator, filename)

    if os.path.isfile(f): # and 'Scrambled' not in f:

        estimator_filelist.append(f)

# --------------------
# Notes: this code is not efficient because it loops more than once over many files. Change this
# ---------------------
#numer of files
n_files = len(file_list)

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#defines the maximum declination
theta_max = np.radians(80)

#-----------------
# directional exposure as aa function of declination
#-----------------
#array to plot theortical directional exposure as a function of declination
dec_cont = np.linspace(-np.pi / 2, np.pi / 2, 5000)

theo_directional_exposure = compute_directional_exposure(dec_cont, theta_max, pao_lat)
integrated_exposure = np.trapz(theo_directional_exposure*np.cos(dec_cont), x=dec_cont)
theo_directional_exposure = (theo_directional_exposure) / integrated_exposure

#save the pdf of the directional exposure for each bin in sin(dec)
bin_centers_omega, bin_content_omega, lower_band_omega, upper_band_omega = data_directional_exposure_dec_func(file_list)

#plot directional exposure from data and theory
fig_exposure_dec = plt.figure(figsize=(5, 4))
ax_exposure_dec = fig_exposure_dec.add_subplot(111)

ax_exposure_dec.plot(bin_centers_omega, bin_content_omega, color = 'tab:blue', label=r'$\langle \omega(\delta) \rangle$ from $10^3$ iso. skies')
ax_exposure_dec.plot(np.sin(dec_cont), theo_directional_exposure, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
ax_exposure_dec.fill_between(bin_centers_omega, lower_band_omega, upper_band_omega, alpha = .5)
ax_exposure_dec = set_style(ax_exposure_dec, '', r'$\sin (\delta)$', r'$\omega(\delta)$', 12)
ax_exposure_dec.legend(loc='upper right', fontsize = 12)

fig_exposure_dec.tight_layout()
fig_exposure_dec.savefig('./results/directional_exposure_dec_func_Theory_%i_IsotropicSkies_th%.0f.pdf' % (n_files, np.degrees(theta_max)))

#------------------------------
# Plot expected number and rate of events
#------------------------------
NSIDE = 128
obs_time = 10 #in years, maybe this should be more flexible
n_events = 1e5 #this should be more flexible
radius_of_target = 1
area_of_target = 2*np.pi*(1 - np.cos(np.radians(radius_of_target)))

#compute theoretical number of events per bin
theo_event_number_per_bin = (.5 / np.pi)*theo_directional_exposure*n_events*area_of_target
theo_event_rate_per_bin = theo_event_number_per_bin / obs_time #in units of per year

#number of events per pixel
bin_centers_event_density, bin_content_event_density, lower_band_event_density, upper_band_event_density = number_of_events_dec_func(estimator_filelist)

#plot directional exposure from data and theory
fig_event_number_dec = plt.figure(figsize=(10, 4))
ax_event_number_dec = fig_event_number_dec.add_subplot(121)
ax_event_rate_dec = fig_event_number_dec.add_subplot(122)

#for event number
ax_event_number_dec.plot(bin_centers_event_density, bin_content_event_density, color = 'tab:blue', label=r'Average from $10^3$ iso. skies')
ax_event_number_dec.plot(np.sin(dec_cont), theo_event_number_per_bin, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
ax_event_number_dec.fill_between(bin_centers_event_density, lower_band_event_density, upper_band_event_density, alpha = .5)
ax_event_number_dec = set_style(ax_event_number_dec, '', r'$\sin (\delta)$', r'Events per target', 12)
ax_event_number_dec.legend(loc='upper right', title = r'$10^5$ events in $10$ yrs. $\psi_{\mathrm{target}} = %.0f^\circ$' % radius_of_target , title_fontsize = 12, fontsize = 12)

#for event rate
ax_event_rate_dec.plot(bin_centers_event_density, bin_content_event_density / obs_time, color = 'tab:blue', label=r'Average from $10^3$ iso. skies')
ax_event_rate_dec.plot(np.sin(dec_cont), theo_event_rate_per_bin, color = 'tab:red', linestyle='dashed', label = 'Theoretical expectation')
ax_event_rate_dec.fill_between(bin_centers_event_density, lower_band_event_density / obs_time, upper_band_event_density / obs_time, alpha = .5)
ax_event_rate_dec = set_style(ax_event_rate_dec, '', r'$\sin (\delta)$', r'$\Gamma \;(\mathrm{yr}^{-1})$', 12)
ax_event_rate_dec.legend(loc='upper right', title = r'$10^5$ events in $10$ yrs. $\psi_{\mathrm{target}} = %.0f^\circ$' % radius_of_target, title_fontsize = 12, fontsize = 12)

fig_event_number_dec.tight_layout()
fig_event_number_dec.savefig('./results/event_rate_per_target_data_theo_%i_IsotropicSkies_th%.0f.pdf' % (n_files, np.degrees(theta_max)))
