import numpy as np
import pandas as pd
import math

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys

sys.path.append('./src/')

from event_manip import time_ordered_events
from event_manip import ang_diff

#compute theta for u uniform in [0, 1]
def compute_dec(u):
    return np.arcsin(2*u - 1)

#compute phi for u uniform in [0, 1]
def compute_ra(u):
    return 2*np.pi*u

#accept events given the auger instantenous acceptance
def compute_accepted_events(time, ra, dec, pao_lat, theta_max):

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

    return time[accepted_indices].gps, ra[accepted_indices], dec[accepted_indices], theta[accepted_indices], lst[accepted_indices]

#fix seed
seed(datetime.now())

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

#define the maximum allowed theta
theta_max = np.radians(80)

#define the number of sources
n_initial_flares = 10_000
n_final_flares = 1_000

#generate vectors of uniformly distributed
rand_a = np.random.random(n_initial_flares)
rand_b = np.random.random(n_initial_flares)

#compute theta and phi for each value of u and v
dec = compute_dec(rand_a)
ra = compute_ra(rand_b)
time = np.random.randint(start_date, end_date, n_initial_flares)
time = Time(time, format='gps', scale='utc', location=pao_loc)

#accept flares
accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = compute_accepted_events(time, ra, dec, lat_pao, theta_max)
accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = accepted_time[:n_final_flares], accepted_ra[:n_final_flares], accepted_dec[:n_final_flares], accepted_theta[:n_final_flares], accepted_lst[:n_final_flares]

#order events by time
ordered_time_indices = accepted_time.argsort()
accepted_time, accepted_ra, accepted_dec = accepted_time[ordered_time_indices], accepted_ra[ordered_time_indices], accepted_dec[ordered_time_indices]

flare_source_data = pd.DataFrame(zip(accepted_time, np.degrees(accepted_ra), np.degrees(accepted_dec), np.degrees(accepted_theta), np.degrees(accepted_lst)), columns=['gps_time_flare', 'ra_flare', 'dec_flare', 'theta_flare', 'lst_flare'])

print(flare_source_data)

flare_source_data.to_parquet('./datasets/MockFlares_%i_%s_%s.parquet' % (int(n_final_flares), start_date_fits, end_date_fits), index=True)