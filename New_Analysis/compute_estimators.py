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

    #convert to declination
    #dec = colat_to_dec(colat)

    return ra, colat

#computes tau for each region of healpy map
def compute_estimators(event_data, NSIDE, rate, target_radius, time_window, theta_max, pao_lat):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_colat = dec_to_colat(event_dec)

    #delete dataframe from memory
    del event_data

    #compute number of pixels
    npix = hp.nside2npix(NSIDE)
    n_events = len(event_dec)
    pixel_area = 4*np.pi / npix

    #get exposure map
    exposure_map = get_normalized_exposure_map(NSIDE, theta_max, pao_lat)

    #array with all pixel indices and event pixel indices
    all_pixel_indices = np.arange(npix)
    all_event_pixel_indices = hp.ang2pix(NSIDE, event_colat, event_ra)

    #save ra and colatitude from center of pixels
    ra_center, colat_center = get_binned_sky_centers(NSIDE, all_pixel_indices)

    #save vector corresponding to center of each pixel
    vec_center = hp.ang2vec(colat_center, ra_center)

    #save pixel indices of each query disc centered at each pixel
    all_pixels_target = np.array([hp.query_disc(NSIDE, vec = vec, radius=target_radius) for vec in vec_center], dtype=object)

    #initialize arrays
    ra_target = []
    colat_target = []
    tau_array = []
    n_max_short_array = []
    n_max_medium_array = []
    n_max_large_array = []
    lambda_array = []
    events_in_target = []
    expected_events_in_target = []
    LiMa_significance_array = []

    for i, pixels_per_target in enumerate(all_pixels_target):

        #ensure all pixels in target are within non-zero exposure
        if np.any(exposure_map[pixels_per_target] == 0):
            continue

        #save corresponding right ascensions and declinations
        ra_target.append(ra_center[i])
        colat_target.append(colat_center[i])

        #event indices corresponding to a given target
        event_pixels_per_target = np.where(np.isin(all_event_pixel_indices, pixels_per_target))[0]

        #compute Li Ma significance for a given target
        exposure_on, exposure_off, LiMa_significance = local_LiMa_significance(n_events, event_pixels_per_target, pixels_per_target, pixel_area, exposure_map, theta_max, pao_lat)

        #times of the events in each target
        times = event_time[event_pixels_per_target]

        if len(event_pixels_per_target) <= 1:
            tau = np.nan
            lambda_estimator = np.nan
            n_max_short = len(event_pixels_per_target)
            n_max_medium = len(event_pixels_per_target)
            n_max_large = len(event_pixels_per_target)

        if len(event_pixels_per_target) > 1:

            #sort events by time
            times.sort()

            #compute time differences
            delta_times = np.diff(times)

            #compute maximum number of events in given time_window
            n_max_short = max([ len(times[np.where( (times < time + time_window[0]) & (times >= time) )[0]]) for time in times ])
            n_max_medium = max([ len(times[np.where( (times < time + time_window[1]) & (times >= time) )[0]]) for time in times ])
            n_max_large = max([ len(times[np.where( (times < time + time_window[2]) & (times >= time) )[0]]) for time in times ])

            #computes tau and lambda
            local_rate = rate*exposure_on

            tau = delta_times[0]*local_rate
            lambda_estimator = -np.sum(np.log(delta_times*local_rate))

        #fill array with estimator values
        tau_array.append(tau)
        events_in_target.append(len(event_pixels_per_target))
        expected_events_in_target.append(exposure_on*n_events)
        lambda_array.append(lambda_estimator)
        n_max_short_array.append(n_max_short)
        n_max_medium_array.append(n_max_medium)
        n_max_large_array.append(n_max_large)
        LiMa_significance_array.append(LiMa_significance)

        if (i % 1000 == 0):
            print(i , '/', len(all_pixels_target), 'targets done!')

    dec_target = colat_to_dec(np.array(colat_target))

    #build dataframe with tau for each pixel in healpy map
    estimator_data = pd.DataFrame(zip(np.degrees(np.array(ra_target)), np.degrees(np.array(dec_target)), events_in_target, expected_events_in_target, tau_array, n_max_short_array, n_max_medium_array, n_max_large_array, lambda_array, LiMa_significance_array), columns = ['ra_center', 'dec_center', 'events_in_target', 'expected_events_in_target', 'tau', 'nMax_1day', 'nMax_1week', 'nMax_1month', 'lambda', 'LiMa_significance'])

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
path_output = './' + path_input.split('/')[1] + '/estimators'

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

#defines the NSIDE parameter
NSIDE = 128

#defines the time window
time_window = [86_164, 7*86_164, 30*86_164] #consider a time window of a single day

#compute event average event rate in angular window
target_radius = np.radians(1)
rate = (n_events / obs_time)

#computing estimators for a given skymap of events
start_time = datetime.now()

estimator_data = compute_estimators(event_data, NSIDE, rate, target_radius, time_window, theta_max, pao_lat)

end_time = datetime.now() - start_time

print('Computing estimators took', end_time,'s')

#order events by declination
estimator_data = estimator_data.sort_values('dec_center')
estimator_data = estimator_data.reset_index(drop=True)

#prints table with summary of data
print(estimator_data)
print(estimator_data.describe())

#saves estimator data
estimator_data.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
