import numpy as np
import pandas as pd
import math

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys
import os

sys.path.append('../src/')

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import compute_directional_exposure

#compute theta for u uniform in [0, 1]
def compute_dec(u, dec_max, dec_min):

    return np.arcsin(u*(np.sin(dec_max) - np.sin(dec_min)) + np.sin(dec_min))

#compute phi for u uniform in [0, 1]
def compute_ra(u, ra_min, ra_max):
    return (ra_max - ra_min)*u + ra_min

#accept events given the auger instantenous acceptance
def compute_accepted_events(time, ra, dec, pao_loc, theta_max):

    #save the pao latitude
    pao_lat = pao_loc.lat.rad

    # compute local sidereal time
    lst = time.sidereal_time('apparent').rad

    # compute zenith angle of event and only accept if smaller than theta_max
    theta = ang_diff(dec, lat_pao, ra, lst)
    accept = theta < theta_max

    time = time[accept]
    ra = ra[accept]
    dec = dec[accept]
    lst = lst[accept]
    theta = theta[accept]

    #compute acceptance probability
    accept_prob = np.cos(theta)

    rand_number = np.random.random(len(time))

    accepted_indices = rand_number < accept_prob

    #compute the azimuth for the accepted events
    equatorial_coords = SkyCoord(ra[accepted_indices]*u.rad, dec[accepted_indices]*u.rad, frame='icrs')
    horizontal_coords = equatorial_coords.transform_to(AltAz(obstime=time[accepted_indices], location=pao_loc))

    azimuth = horizontal_coords.az.rad

    return time[accepted_indices].gps, ra[accepted_indices], dec[accepted_indices], theta[accepted_indices], azimuth, lst[accepted_indices]

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

#define the width of the patch in sky
width_patch = np.radians(10)

#define the position of a source
dec_source = 0
ra_source = 0

#define the number of events
n_events = 100_000

#define the maximum accepted zenith angle
theta_max = np.radians(80)
all_dec_max = lat_pao + theta_max

dec_min = dec_source - width_patch
dec_max = dec_source + width_patch
ra_min = ra_source - width_patch
ra_max = ra_source + width_patch

if dec_max > all_dec_max:
    print('please make sure all events in patch are in FoV of observatory!')
    exit()

#compute the integrated exposure over the circular area to determine the number of expected events in the cap
dec_in_patch = np.linspace(dec_min, dec_max, 1000)
dec_range = np.linspace(-np.pi / 2, all_dec_max, 10000)

exposure_in_patch = compute_directional_exposure(dec_in_patch, theta_max, lat_pao)
total_exposure = compute_directional_exposure(dec_range, theta_max, lat_pao)

integrated_exposure_in_patch = 2*width_patch*np.trapz(exposure_in_patch*np.cos(dec_in_patch), x = dec_in_patch)
total_integrated_exposure = 2*np.pi*np.trapz(total_exposure*np.cos(dec_range), x = dec_range)

n_events_accepted_in_patch = np.ceil((integrated_exposure_in_patch / total_integrated_exposure)*n_events).astype('int')

#multiply the computed number of events by factor that takes into account the rejection of events
print(n_events_accepted_in_patch)

n_events_in_patch = int(10*n_events_accepted_in_patch)

#generate vectors of uniformly distributed
rand_a = np.random.random(n_events_in_patch)
rand_b = np.random.random(n_events_in_patch)

#compute theta and phi for each value of u and v
dec = compute_dec(rand_a, dec_min, dec_max)
ra = compute_ra(rand_b, ra_min, ra_max)

time = np.random.randint(start_date, end_date, n_events_in_patch)
time = Time(time, format='gps', scale='utc', location=pao_loc)

#accept events and save first few accepted events in file
start = datetime.now()

accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = compute_accepted_events(time, ra, dec, pao_loc, theta_max)

print('Efficiency in accepting events =', len(accepted_time) / n_events_in_patch)
print('This took ', datetime.now() - start, ' s')

#accept only the first n_events_accepted_in_patch events
accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = accepted_time[:n_events_accepted_in_patch], accepted_ra[:n_events_accepted_in_patch], accepted_dec[:n_events_accepted_in_patch], accepted_theta[:n_events_accepted_in_patch], accepted_phi[:n_events_accepted_in_patch], accepted_lst[:n_events_accepted_in_patch]

#order events by time
accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst)

accepted_event_data = pd.DataFrame(zip(accepted_time, np.degrees(accepted_ra), np.degrees(accepted_dec), np.degrees(accepted_theta), np.degrees(accepted_phi), np.degrees(accepted_lst)), columns=['gps_time', 'ra', 'dec', 'theta', 'phi', 'lst'])

print(accepted_event_data)

#define the output path
output_path = './datasets'

if not os.path.exists(output_path):
    os.makedirs(output_path)

accepted_event_data.to_parquet('./datasets/IsoDist_%i_decCenter_%i_raCenter_%i_patchWidth_%.0f_acceptance_th80_10years.parquet' % (int(n_events_accepted_in_patch), dec_source, ra_source, np.degrees(width_patch)), index=True)
