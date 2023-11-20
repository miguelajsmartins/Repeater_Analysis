import numpy as np
import pandas as pd
import healpy as hp
import math

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys
import os

sys.path.append('./src/')

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map
from event_manip import local_LiMa_significance
from event_manip import unbinned_local_LiMa_significance

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#converts declination into colat
def dec_to_colat(dec):
    return np.pi / 2 - dec

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE, bin_indexes):

    #get colatidude and ra of each bin center in healpy sky
    colat, ra = hp.pix2ang(NSIDE, bin_indexes)

    return ra, colat

#computes tau for each region of healpy map
def compute_estimators(event_data, rate, target_radius, time_window, theta_max, pao_lat):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())

    #delete dataframe from memory
    del event_data

    #eliminate events such that region of target lies outside observed sky
    dec_max = theta_max + pao_lat
    in_observed_sky = (event_dec + target_radius < dec_max)

    event_time = event_time[in_observed_sky]
    event_ra = event_ra[in_observed_sky]
    event_dec = event_dec[in_observed_sky]

    #compute the target area
    area_of_target = 2*np.pi*(1 - np.cos(target_radius))

    #get compute directional exposure for the entire sky
    dec = np.linspace(-np.pi / 2, np.pi / 2, 5000)
    exposure_fullsky = compute_directional_exposure(dec, theta_max, pao_lat)
    integrated_exposure = 2*np.pi*np.trapz(exposure_fullsky*np.cos(dec), x=dec)
    exposure_fullsky = exposure_fullsky / integrated_exposure

    #initialize arrays
    tau_array = []
    n_max_short_array = []
    n_max_medium_array = []
    n_max_large_array = []
    lambda_array = []
    events_in_target = []
    expected_events_in_target = []
    LiMa_significance_array = []

    for i in range(len(event_dec)):

        #save dec and right ascension of event in center of each target
        dec_target = event_dec[i]
        ra_target = event_ra[i]

        #exclude events based on declination and right ascension
        in_strip = (np.abs(dec_target - event_dec) < target_radius) #& (np.abs(ra_target - event_ra) < target_radius)

        events_in_strip_dec = event_dec[in_strip]
        events_in_strip_ra = event_ra[in_strip]
        events_in_strip_time = event_time[in_strip]

        #compute the angular difference between center of each target and all other events within target radius
        ang_diffs = ang_diff(dec_target, events_in_strip_dec, ra_target, events_in_strip_ra)

        #keep only events in target
        event_indices_in_target = (ang_diffs < target_radius) & (ang_diffs != 0)

        #save the arrival times of the events in the target region
        events_in_target_time = events_in_strip_time[event_indices_in_target]

        if ( (np.abs(np.degrees(dec_target) + 27.5) < 1) and (np.abs(np.degrees(ra_target) - 181.5) < 1)) :
            print('dec_target =', np.degrees(dec_target))
            print('events in target', events_in_target_time.shape)

        #compute Li Ma significance for a given target
        exposure_on, exposure_off, LiMa_significance = unbinned_local_LiMa_significance(n_events, events_in_target_time, area_of_target, integrated_exposure, dec_target, theta_max, pao_lat)

        if ( (np.abs(np.degrees(dec_target) + 27.5) < 1) and (np.abs(np.degrees(ra_target) - 181.5) < 1)) :
            print('expected_events_in_target', exposure_on*n_events)
            print('Li Ma significance', LiMa_significance)


        if len(events_in_target_time) <= 1:
            tau = np.nan
            lambda_estimator = np.nan
            n_max_short = len(events_in_target_time)
            n_max_medium = len(events_in_target_time)
            n_max_large = len(events_in_target_time)

        else:

            #sort events by time
            events_in_target_time.sort()

            #compute time differences
            delta_times = np.diff(events_in_target_time)

            #compute maximum number of events in given time_window
            n_max_short = max([ len(events_in_target_time[np.where( (events_in_target_time < time + time_window[0]) & (events_in_target_time >= time) )[0]]) for time in events_in_target_time])
            n_max_medium = max([ len(events_in_target_time[np.where( (events_in_target_time < time + time_window[1]) & (events_in_target_time >= time) )[0]]) for time in events_in_target_time])
            n_max_large = max([ len(events_in_target_time[np.where( (events_in_target_time < time + time_window[2]) & (events_in_target_time  >= time) )[0]]) for time in events_in_target_time])

            #computes tau and lambda
            local_rate = rate*exposure_on
            tau = delta_times[0]*local_rate
            lambda_estimator = -np.sum(np.log(delta_times*local_rate))

        #fill array with estimator values
        tau_array.append(tau)
        events_in_target.append(len(events_in_target_time))
        expected_events_in_target.append(exposure_on*n_events)
        lambda_array.append(lambda_estimator)
        n_max_short_array.append(n_max_short)
        n_max_medium_array.append(n_max_medium)
        n_max_large_array.append(n_max_large)
        LiMa_significance_array.append(LiMa_significance)

        if (i % 1000 == 0):
            print(i , '/', len(event_dec), 'targets done!')

    #build dataframe with tau for each pixel in healpy map
    estimator_data = pd.DataFrame(zip(np.degrees(event_ra), np.degrees(event_dec), events_in_target, expected_events_in_target, tau_array, n_max_short_array, n_max_medium_array, n_max_large_array, lambda_array, LiMa_significance_array), columns = ['ra_center', 'dec_center', 'events_in_target', 'expected_events_in_target', 'tau', 'nMax_1day', 'nMax_1week', 'nMax_1month', 'lambda', 'LiMa_significance'])

    return estimator_data

#load events from isotropic distribution
if (len(sys.argv) == 1):
    print('Must give a file containing distribution of events')
    exit()

#save file name
filename = sys.argv[1]

#checks if file exists
if not os.path.exists(filename):
    print("Requested file does not exist. Aborting")
    exit()

#save path to file and its basename
path_input = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]
#path_output = './' + path_input.split('/')[1] + '/estimators'

#save dataframe with isotropic events
event_data = pd.read_parquet(filename, engine = 'fastparquet')

#compute number of events and observation time
n_events = len(event_data.index)
obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#defines the maximum declination and a little tolerance
theta_max = np.radians(80)
dec_max = pao_lat + theta_max

#defines the time window
time_window = [86_164, 7*86_164, 30*86_164] #consider a time window of a single day

#compute event average event rate in angular window
target_radius = np.radians(1)
rate = (n_events / obs_time)

#computing estimators for a given skymap of events
start_time = datetime.now()

estimator_data = compute_estimators(event_data, rate, target_radius, time_window, theta_max, pao_lat)

end_time = datetime.now() - start_time

print('Computing estimators took', end_time,'s')

#order events by declination
estimator_data = estimator_data.sort_values('dec_center')
estimator_data = estimator_data.reset_index(drop=True)

#prints table with summary of data
print(estimator_data)
print(estimator_data.describe())

#saves estimator data
estimator_data.to_parquet(path_input + '/' + basename + '_Estimators.parquet', index = True)
