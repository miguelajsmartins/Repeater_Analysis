import numpy as np
import pandas as pd
import math

import numpy.ma as ma

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

#compute theta for u uniform in [0, 1]
def compute_dec(u, dec_max):
    return np.arcsin(u*(1 + np.sin(dec_max)) - 1)

#compute phi for u uniform in [0, 1]
def compute_ra(u):
    return 2*np.pi*u

#accept events given the auger instantenous acceptance
def compute_accepted_events(time, ra, dec, pao_lat, theta_max):

    # compute local sidereal time
    lst = time.sidereal_time('mean').rad
    time = time.gps

    #compute theta
    theta = ang_diff(dec, lat_pao, ra, lst)

    #define the acceptance probability
    accept_prob = np.cos(theta)
    rand_number = np.random.random(time.shape)

    #accept events
    accept = np.logical_and(theta < theta_max, rand_number < accept_prob)

    #mask arrays
    time = ma.masked_array(time, mask = np.logical_not(accept)).filled(fill_value = np.nan)
    ra = ma.masked_array(ra, mask = np.logical_not(accept)).filled(fill_value = np.nan)
    dec = ma.masked_array(dec, mask = np.logical_not(accept)).filled(fill_value = np.nan)
    lst = ma.masked_array(lst, mask = np.logical_not(accept)).filled(fill_value = np.nan)
    theta = ma.masked_array(theta, mask = np.logical_not(accept)).filled(fill_value = np.nan)

    return time, ra, dec, theta, lst

#fix seed
seed = 47
np.random.seed(seed)

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#define start and end times
start_date_fits = '2010-01-01'
end_date_fits = '2020-01-01'
start_date = Time(start_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps
end_date = Time(end_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps

#define the number of events
n_events = 400_000
n_accept = int(n_events / 4)

#define the maximum accepted zenith angle
theta_max = np.radians(80)
dec_max = lat_pao + theta_max

#define the number of samples
n_samples = int(10)

#generate vectors of uniformly distributed
rand_a = np.random.random((n_samples, n_events))
rand_b = np.random.random((n_samples, n_events))

#compute theta and phi for each value of u and v
dec = compute_dec(rand_a, dec_max)
ra = compute_ra(rand_b)

time = (end_date - start_date)*np.random.random((n_samples, n_events)) + start_date

time = Time(time, format='gps', scale='utc', location=pao_loc)

#accept events and save first few accepted events in file
start_1 = datetime.now()

accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = compute_accepted_events(time, ra, dec, lat_pao, theta_max)

start_2 = datetime.now()

print('Accepting events', start_2 - start_1, ' s')

#time order events
#accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst)

#define the output path
output_path = './datasets/iso_samples'

for sample in range(n_samples):

    #save the data as a dataframe
    accepted_event_data = pd.DataFrame(zip(accepted_time[sample,:], np.degrees(accepted_ra[sample,:]), np.degrees(accepted_dec[sample,:]), np.degrees(accepted_theta[sample,:]), np.degrees(accepted_lst[sample,:])), columns=['gps_time', 'ra', 'dec', 'theta', 'lst'])

    #drop nan values
    accepted_event_data.dropna(inplace = True, ignore_index = True)

    #select only n_accept events
    accepted_event_data = accepted_event_data.sample(n = n_accept, ignore_index = True)

    #order events by time
    accepted_event_data.sort_values(by = 'gps_time', inplace = True, ignore_index = True)

    print(accepted_event_data)

    #save dataframe
    accepted_event_data.to_parquet(os.path.join(output_path, 'IsoDist_%i_acceptance_th80_%s_%s_sample_%i.parquet' % (int(n_accept), start_date_fits, end_date_fits, sample)), index=True)

print('Cleaning and saving samples took', datetime.now() - start_2, ' s')
