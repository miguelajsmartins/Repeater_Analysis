import pandas as pd
import numpy as np
import healpy as hp
import math
from healpy.newvisufunc import projview, newprojplot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

import scipy.interpolate as spline
from scipy.stats import poisson

sys.path.append('./src/')
sys.path.append('.')

import hist_manip
from hist_manip import data_2_binned_errorbar
import event_manip

from event_manip import compute_directional_exposure
from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import get_normalized_exposure_map
from event_manip import get_skymap
from event_manip import colat_to_dec

from axis_style import set_style

from array_manip import unsorted_search

from compute_lambda_pvalue_binned_sky import compute_p_value

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#convert alpha into phi for ploting
def ra_to_phi(ra):
    return np.where(ra > np.pi, ra - 2*np.pi, ra)

#file to get the positions of sources given an events file
def get_sources_position(event_file):

    event_data = pd.read_parquet(event_file, engine='fastparquet')

    #get total number of events
    n_events = len(event_data.index)

    #restrict to events with explosions
    event_data = event_data[event_data['is_from_flare'] == True]

    #save the colatitude and right ascension of each explosion
    colat = event_manip.dec_to_colat(np.radians(event_data['flare_dec'].to_numpy()))
    ra = np.radians(event_data['flare_ra'].to_numpy())

    ordered_indices = colat.argsort()
    colat, ra = colat[ordered_indices], ra[ordered_indices]

    colat_unique = np.unique(colat)
    ra_unique = ra[unsorted_search(colat, colat_unique)]

    return n_events, colat_unique, ra_unique

#compute the unpenalized and penalized p values for lambda and poisson
def compute_targeted_search_penalized_pvalues(event_data, flare_dec, flare_ra, lambda_dist_data, target_radius, rate, theta_max, pao_lat):

    #define declination and right ascensions of targets
    target_dec = flare_dec
    target_ra = flare_ra

    #save declination, right ascension and time of events
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_time = event_data['gps_time'].to_numpy()

    n_events = len(event_dec)

    #delete dataframe
    del event_data

    #define area of each target
    target_area = 2*np.pi*(1 - np.cos(target_radius))

    #get directional exposure from entire sky
    dec = np.linspace(-np.pi / 2, np.pi / 2, 5000)
    exposure_fullsky = compute_directional_exposure(dec, theta_max, pao_lat)
    integrated_exposure = 2*np.pi*np.trapz(exposure_fullsky*np.cos(dec), x=dec)

    #initialize lists
    lambda_array = []
    events_in_target_array = []
    expected_events_in_target_array = []

    for i in range(len(target_dec)):

        #save dec and right ascension of event in center of each target
        dec_center = target_dec[i]
        ra_center = target_ra[i]

        #exclude events based on declination and right ascension
        in_strip = (np.abs(dec_center - event_dec) < target_radius) #& (np.abs(ra_target - event_ra) < target_radius)

        events_in_strip_dec = event_dec[in_strip]
        events_in_strip_ra = event_ra[in_strip]
        events_in_strip_time = event_time[in_strip]

        #compute the angular difference between center of each target and all other events within target radius
        ang_diffs = ang_diff(dec_center, events_in_strip_dec, ra_center, events_in_strip_ra)

        #keep only events in target
        event_indices_in_target = (ang_diffs < target_radius) & (ang_diffs != 0)

        #save the arrival times of the events in the target region
        events_in_target_time = events_in_strip_time[event_indices_in_target]

        #save the actual and expected number of events in each target
        events_in_target = len(events_in_target_time)
        exposure_on = target_area*(compute_directional_exposure([dec_center], theta_max, pao_lat) / integrated_exposure)[0]
        expected_events_in_target = n_events*exposure_on

        if events_in_target <= 1:
            lambda_estimator = np.nan

        else:

            #compute time differences
            delta_times = np.diff(events_in_target_time)

            #compute local rate and lambda
            local_rate = rate*exposure_on
            lambda_estimator = -np.sum(np.log(delta_times*local_rate))

        #fill array with estimator values
        events_in_target_array.append(events_in_target)
        expected_events_in_target_array.append(expected_events_in_target)
        lambda_array.append(lambda_estimator)

    #convert lists to arrays
    events_in_target_array = np.array(events_in_target_array)
    expected_events_in_target_array = np.array(expected_events_in_target_array)
    lambda_array = np.array(lambda_array)

    #compute poisson p values
    poisson_pvalue = 1 - .5*(poisson.cdf(events_in_target_array - 1, expected_events_in_target_array) + poisson.cdf(events_in_target_array, expected_events_in_target_array))

    #build dataframe with tau for each pixel in healpy map
    target_data = pd.DataFrame(zip(np.degrees(target_ra), np.degrees(target_dec), events_in_target_array, expected_events_in_target_array, lambda_array, poisson_pvalue), columns = ['ra_target', 'dec_target', 'events_in_target', 'expected_events_in_target', 'lambda', 'poisson_p_value'])

    #compute lambda pvalue
    target_data['lambda_p_value'] = target_data.apply(lambda x: compute_p_value(lambda_dist_data, x['lambda'], x['expected_events_in_target']) if not math.isnan(x['lambda']) else np.nan, axis = 1)

    #penalize p-values by the number of trials
    n_trials = len(target_dec)
    target_data['poisson_penalized_pvalue'] = 1 - np.power(1 - target_data['poisson_p_value'], n_trials)
    target_data['lambda_penalized_pvalue'] = 1 - np.power(1 - target_data['lambda_p_value'], n_trials)

    print(target_data.head(len(target_dec)))

    return target_data


#----------------------------------------------------
# Plots for the proceedings of ICRC 2023
#----------------------------------------------------
#complains if no filename is given
if len(sys.argv) < 2:
    print('No filename was given!')
    exit()

#load data with the skymap postrial p values
postrial_pvalue_file = sys.argv[1]

#construct name of file given the name of file read
output_path = os.path.dirname(postrial_pvalue_file)
basename = os.path.splitext(os.path.basename(postrial_pvalue_file))[0]
split_name = basename.split('_')
split_path = output_path.split('/')

output_name_poisson = 'Skymap_' + split_name[0] + '_' + split_name[1] + '_pvalues_Poisson_' + split_name[3] + '_' + split_path[3] + '_nSide_' + split_name[-1] + '.pdf'
output_name_lambda = 'Skymap_' + split_name[0] + '_' + split_name[1] + '_pvalues_Lambda_' + split_name[3] + '_' + split_path[3] + '_nSide_' + split_name[-1] + '.pdf'

#get sample with flare events
event_file = os.path.join(output_path, '_'.join(split_name[2:-3]) + '.parquet')
event_data = pd.read_parquet(event_file, engine='fastparquet')

#load file with pos-trial pvalues
postrial_pvalue_data = pd.read_parquet(postrial_pvalue_file, engine='fastparquet')

#define position of the observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#define the maximum value of theta
theta_max = np.radians(80)

#compute exposure map to define excluded regions
target_radius = np.radians(1)
target_area = 2*np.pi*(1 - np.cos(target_radius))

#load positions of sources
total_events, flare_colat, flare_ra = get_sources_position(event_file)
obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]

rate = total_events / obs_time

#compute penalized p-values for the targets
lambda_dist_file = './datasets/estimator_dist/Lambda_dist_per_rate_997.json'
lambda_dist_data = pd.read_json(lambda_dist_file)

targeted_search_pvalue_data = compute_targeted_search_penalized_pvalues(event_data, colat_to_dec(flare_colat), flare_ra, lambda_dist_data, target_radius, rate, theta_max, lat_pao)

#define the output name of p-values
output_targeted_search = 'TargetedSearch_Penalized_' + '_'.join(split_name[2:]) + '.csv'
output_targeted_search = os.path.join(output_path, output_targeted_search)

#save output of targeted search as csv for incorporate in latex
targeted_search_pvalue_data.to_csv(output_targeted_search, index = True)

#save the target centers and postrial pvalues
target_ra = np.radians(postrial_pvalue_data['ra_target'].to_numpy())
target_colat = event_manip.dec_to_colat(np.radians(postrial_pvalue_data['dec_target'].to_numpy()))
target_event_expectation = postrial_pvalue_data['expected_events_in_target'].to_numpy()
target_poisson_postrial = postrial_pvalue_data['poisson_postrial_p_value'].to_numpy()
target_lambda_postrial = postrial_pvalue_data['lambda_postrial_p_value'].to_numpy()

#delete dataframe from memory
del postrial_pvalue_data

#read the value of the nside, number of flares, flare duration and events per flare parameter of used in the file
NSIDE=int(split_name[-1])
n_flares = int(split_name[9])
n_events_per_flare = int(split_name[11])
flare_duration = int(split_name[13])

#produce skymap with pvalues
npix = hp.nside2npix(NSIDE)

skymap_postrial_poisson = np.zeros(npix)
skymap_postrial_lambda = np.zeros(npix)

pixel_indices = hp.ang2pix(NSIDE, target_colat, target_ra)

#if postrial probabilties are null, then set them to their minimum number
target_poisson_postrial[target_poisson_postrial == 0] = 1/991 #where 991 is the number of samples
target_lambda_postrial[target_lambda_postrial == 0] = 1/991

np.add.at(skymap_postrial_poisson, pixel_indices, np.log10(target_poisson_postrial))
np.add.at(skymap_postrial_lambda, pixel_indices, np.log10(target_lambda_postrial))

#save the color map
color_map = cm.get_cmap('coolwarm').reversed()



rate_map = total_events*target_area*get_normalized_exposure_map(NSIDE, theta_max, lat_pao)

#exclude pixels outside FoV of observatory
low_rate = (rate_map < 1)

skymap_postrial_poisson[low_rate] = hp.UNSEEN
skymap_postrial_poisson = hp.ma(skymap_postrial_poisson)

skymap_postrial_lambda[low_rate] = hp.UNSEEN
skymap_postrial_lambda = hp.ma(skymap_postrial_lambda)

#save figure
fig_skymap = plt.figure(figsize=(10,8)) #create figure

#plot sky map for poisson
hp.newvisufunc.projview(
    skymap_postrial_poisson,
    override_plot_properties={'figure_size_ratio' : .6},
    graticule=True,
    graticule_labels=True,
    title=r"$n_{\mathrm{flares}} = %i$, $n_{\mathrm{events}} = %i$, $\Delta t_{\mathrm{flare}} = %i$ days" % (n_flares, n_events_per_flare, flare_duration),
    xlabel=r"$\alpha$",
    ylabel=r"$\delta$",
    cmap=color_map,
    cb_orientation="horizontal",
    projection_type="hammer",
    fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
    longitude_grid_spacing = 30,
    latitude_grid_spacing = 30,
    xtick_label_color='black',
    min=-3,
    max=0,
    unit = r'$\log_{10} (p_{\mathrm{poisson}}\mathrm{-value})$',
);

hp.newvisufunc.newprojplot(theta=flare_colat, phi=ra_to_phi(flare_ra), marker='o', linestyle = 'None', fillstyle='none', color = 'black', markersize=10)

plt.savefig(os.path.join(output_path, output_name_poisson), dpi=1000)

#plot skymap for lambda
hp.newvisufunc.projview(
    skymap_postrial_lambda,
    override_plot_properties={'figure_size_ratio' : .6},
    graticule=True,
    graticule_labels=True,
    title=r"$n_{\mathrm{flares}} = %i$, $n_{\mathrm{events}} = %i$, $\Delta t_{\mathrm{flare}} = %i$ days" % (n_flares, n_events_per_flare, flare_duration),
    xlabel=r"$\alpha$",
    ylabel=r"$\delta$",
    cmap=color_map,
    cb_orientation="horizontal",
    projection_type="hammer",
    fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
    longitude_grid_spacing = 30,
    latitude_grid_spacing = 30,
    xtick_label_color='black',
    min=-3,
    max=0,
    unit = r'$\log_{10} (p_{\Lambda} \mathrm{-value})$',
);

newprojplot(theta=flare_colat, phi=ra_to_phi(flare_ra), marker='o', linestyle = 'None', fillstyle='none', color = 'black', markersize=10)

plt.savefig(os.path.join(output_path, output_name_lambda), dpi=1000)
