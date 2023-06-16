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

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#converts declination into colat
def dec_to_colat(dec):
    return np.pi / 2 - dec

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE):

    #compute the number of pixels
    npix = hp.nside2npix(NSIDE)

    #get array with bin indexes
    bin_indexes = np.arange(npix)

    #get colatidude and ra of each bin center in helpy sky
    colat, ra = hp.pix2ang(NSIDE, bin_indexes)

    #convert to declination
    dec = colat_to_dec(colat)

    return ra, dec


#computes tau for each region of healpy map
def compute_estimators(event_data, NSIDE, rate, time_window, theta_max, pao_lat): #ra_center, dec_center, ang_window, time_window, pao_loc, rate):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_colat = dec_to_colat(event_dec)

    #delete dataframe from memory
    del event_data

    #compute number of pixels
    npix = hp.nside2npix(NSIDE)

    #get exposure map
    exposure_map = get_normalized_exposure_map(NSIDE, theta_max, pao_lat)

    #array with all pixel indices and event pixel indices
    all_pixel_indices = np.arange(npix)
    all_event_pixel_indices = hp.ang2pix(NSIDE, event_colat, event_ra)

    #tau array
    tau_array = []
    n_max_array = []
    lambda_array = []

    for i, pixel_index in enumerate(all_pixel_indices):

        #event indices corresponding to a given pixel
        event_pixels = np.where(all_event_pixel_indices == pixel_index)[0]

        if len(event_pixels) == 0 or len(event_pixels) == 1:
            tau = np.nan
            lambda_estimator = np.nan
            n_max = len(event_pixels)

        if len(event_pixels) > 1:

            #save events within that pixel
            new_event_dec = event_dec[event_pixels]
            new_event_ra = event_ra[event_pixels]
            times = event_time[event_pixels]

            #computes exposure ratio
            exposure_on = exposure_map[pixel_index]
            exposure_off = np.copy(exposure_map)
            exposure_off = np.delete(exposure_off, pixel_index)
            exposure_off = sum(exposure_off)

            alpha = exposure_on / exposure_off

            #sort events by time
            times.sort()

            #compute time differences
            delta_times = np.diff(times)

            #compute maximum number of events in given time_window
            n_max = max([ len(times[np.where( (times < time + time_window) & (times > time) )[0]]) for time in times ])

            #computes tau and lambda
            local_rate = rate*alpha

            tau = delta_times[0]*local_rate
            lambda_estimator = -np.sum(np.log(delta_times*local_rate))

            if len(event_pixels) > 10:
                print(delta_times*local_rate)
                print(lambda_estimator)

        #fill array with estimator values
        tau_array.append(tau)
        lambda_array.append(lambda_estimator)
        n_max_array.append(n_max)

    #convert lists to vectors
    tau_array = np.array(tau_array)
    n_max_array = np.array(n_max_array)
    lambda_array = np.array(lambda_array)

    #save ra and declinations from center of pixels
    ra_center, dec_center = get_binned_sky_centers(NSIDE)

    #build dataframe with tau for each pixel in healpy map
    estimator_data = pd.DataFrame(zip(np.degrees(ra_center), np.degrees(dec_center), np.array(tau_array), n_max_array, lambda_array), columns = ['ra_center', 'dec_center', 'tau', 'nMax_1day', 'lambda'])

    return estimator_data

#computes the Li MA significance for each pixel
def compute_LiMa_significance(event_data, NSIDE, theta_max, pao_lat):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_colat = dec_to_colat(event_dec)

    #delete dataframe from memory
    del event_data

    #saves number of events
    n_events = len(event_colat)

    #compute number of pixels
    npix = hp.nside2npix(NSIDE)

    #get exposure map
    exposure_map = get_normalized_exposure_map(NSIDE, theta_max, pao_lat)

    #array with all pixel indices and event pixel indices
    all_pixel_indices = np.arange(npix)
    all_event_pixel_indices = hp.ang2pix(NSIDE, event_colat, event_ra)

    #initialize significance
    significance_array = []

    for i, pixel_index in enumerate(all_pixel_indices):

        #event indices corresponding to a given pixel
        event_pixels = np.where(all_event_pixel_indices == pixel_index)[0]

        if len(event_pixels) == 0:
            significance = np.nan

        if len(event_pixels) > 1:

            #computes n_on and n_off
            N_on = len(event_pixels)
            N_off = n_events - N_on
            ratio_on = N_on / n_events
            ratio_off = N_off / n_events

            #computes LiMa alpha given exposures
            exposure_on = exposure_map[pixel_index]
            exposure_off = np.copy(exposure_map)
            exposure_off = np.delete(exposure_off, pixel_index)
            exposure_off = sum(exposure_off)

            alpha = exposure_on / exposure_off

            #compute significance
            parcel_on = N_on * np.log((1 + 1 / alpha)*ratio_on)
            parcel_off = N_off * np.log((1 + alpha)*ratio_off)

            significance = np.sqrt(2)*np.sign(N_on - alpha*N_off)*np.sqrt(parcel_on + parcel_off)

        significance_array.append(significance)

    return significance_array


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

#event_data = event_data.drop(labels=range(10_000, len(event_data.index)), axis = 0)

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#defines the maximum declination
theta_max = np.radians(80)
dec_max = pao_lat + theta_max

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#defines the NSIDE parameter
NSIDE = 64

#defines the time window
time_window = 86_164 #consider a time window of a single day

#compute event average event rate in angular window
ang_window = hp.nside2resol(NSIDE)
rate = (n_events / obs_time)

#computing estimators for a given skymap of events
start_time = datetime.now()

estimator_data = compute_estimators(event_data, NSIDE, rate, time_window, theta_max, pao_lat)

end_time = datetime.now() - start_time

print('Computing estimators took', end_time,'s')

#computing Li Ma significance
start_time_2 = datetime.now()

LiMa_significance = compute_LiMa_significance(event_data, NSIDE, theta_max, pao_lat)

end_time_2 = datetime.now() - start_time_2

print('Computing LiMa significance took', end_time,'s')

#add Li Ma significance to estimator data
estimator_data['LiMa_significance'] = pd.Series(LiMa_significance)

#order events by declination
estimator_data = estimator_data.sort_values('dec_center')
estimator_data = estimator_data.reset_index(drop=True)

#prints table with summary of data
print(estimator_data.describe())

#saves estimator data
estimator_data.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
