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
from event_manip import time_ordered_events_ra_dec

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE, dec_max):

    #compute the number of pixels
    npix = hp.nside2npix(NSIDE)

    #get array with bin indexes
    bin_indexes = np.arange(npix)

    #get colatidude and ra of each bin center in helpy sky
    colat, ra = hp.pix2ang(NSIDE, bin_indexes)

    #convert to declination
    dec = colat_to_dec(colat)

    accept = dec < dec_max

    return ra[accept], dec[accept]


#computes tau for each region of healpy map
def compute_estimators(event_data, ra_center, dec_center, ang_window, time_window, pao_loc, rate):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())

    fixed_event_ra = event_ra
    fixed_event_dec = event_dec

    #delete dataframe from memory
    del event_data

    #tau array
    tau_array = []
    n_max_array = []
    lambda_array = []

    for i in range(len(ra_center)):

        #only consider events such that declination and ra are both below angular_window
        inside_square = (np.abs(event_dec - dec_center[i]) < ang_window) & (np.abs(ra_center[i] - event_ra) < ang_window)

        new_event_dec = event_dec[inside_square]
        new_event_ra = event_ra[inside_square]
        new_event_time = event_time[inside_square]

        #save number of events
        n_events = len(new_event_dec)

        #ensure that ra_center and dec_center are of the same size as event_dec
        vec_with_ones = np.ones(n_events)
        dec_center_vec = dec_center[i]*vec_with_ones
        ra_center_vec = ra_center[i]*vec_with_ones

        #computes angular difference
        psi = ang_diff(dec_center_vec, new_event_dec, ra_center_vec, new_event_ra)

        #select events such that they are seperated by less than ang_window
        is_in_ang_window = psi < ang_window

        times = new_event_time[is_in_ang_window]

        if len(times) < 2:
            tau = np.nan
            n_max = np.nan
            lambda_estimator = np.nan
        else:

            times.sort()

            delta_times = np.diff(times)

            n_max = max([len(times[np.where(times < time + time_window)[0]]) for time in times])

            #computes tau
            tau = delta_times[0]
            lambda_estimator = len(delta_times)*np.log(rate) + sum(np.log(delta_times))

        #fill array with tau values
        tau_array.append(tau)
        lambda_array.append(lambda_estimator)
        n_max_array.append(n_max)

        #print(i, 'events done')

    #build dataframe with tau for each pixel in healpy map
    tau_data = pd.DataFrame(zip(np.degrees(dec_center), np.degrees(ra_center), np.array(tau_array)*rate, n_max_array, lambda_array), columns = ['ra_center', 'dec_center', 'tau', 'nMax_1day', 'lambda'])

    return tau_data

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
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#defines the maximum declination
theta_max = np.radians(80)
dec_max = lat_pao + theta_max

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#defines the NSIDE parameter
NSIDE = 128

#defines the angular window
ang_window = np.radians(1)
time_window = 86_164 #consider a time window of a single day

#compute event average event rate in angular window
rate = (n_events / obs_time)*((1 - np.cos(ang_window)) / (1 + np.sin(dec_max)))

#gets the bin_centers of the healpy sky
ra_center, dec_center = get_binned_sky_centers(NSIDE, dec_max)

#computing tau
start_time = datetime.now()

estimator_data = compute_estimators(event_data, ra_center, dec_center, ang_window, time_window, pao_loc, rate)

print('Computing estimators took', datetime.now() - start_time ,'s')

print(estimator_data.count())

estimator_data.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
