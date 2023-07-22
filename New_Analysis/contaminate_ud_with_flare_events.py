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

sys.path.append('./src/')

from event_manip import time_ordered_events
from event_manip import ang_diff
from array_manip import unsorted_search

#compute phi for u uniform in [0, 1]
def compute_phi(u):
    return 2*np.pi*u

#convert zenith to altitude
def zenith_to_alt(theta):
    return np.pi/2 - theta

#scramble events by suffling event sidereal time and theta independently, and sampling phi from uniform dist
def get_flare_events(flare_data, flare_duration, initial_events_per_flare, final_events_per_flare, resolution, seed, pao_loc, theta_max):

    #save coordinates of flare
    flare_dec = np.radians(flare_data['dec_flare'].to_numpy())
    flare_ra = np.radians(flare_data['ra_flare'].to_numpy())

    #define the beginning and end of flare
    flare_begin = flare_data['gps_time_flare'].to_numpy()
    flare_end = flare_begin + flare_duration

    #define seed
    np.random.seed(seed)

    #select, with uniform dist, time stamps for events, and angular positions following a gaussian distribution
    event_times = np.concatenate([np.random.randint(flare_begin[i], flare_end[i], initial_events_per_flare) for i in range(len(flare_begin))])
    event_dec = np.concatenate([np.random.normal(flare_dec[i], resolution, initial_events_per_flare) for i in range(len(flare_begin))])
    event_ra = np.concatenate([np.random.normal(flare_ra[i], resolution, initial_events_per_flare) for i in range(len(flare_begin))])
    event_lst = Time(event_times, format='gps', scale='utc', location=pao_loc).sidereal_time('apparent').rad
    event_theta = ang_diff(event_dec, pao_loc.lat.rad, event_ra, event_lst)

    #exclude flare events outside FoV of observatory
    inside_fov = event_theta < theta_max
    event_times, event_ra, event_dec, event_theta, event_lst = event_times[inside_fov], event_ra[inside_fov], event_dec[inside_fov], event_theta[inside_fov], event_lst[inside_fov]

    #save only final_events_per_flare from each flare
    accepted_event_times = np.concatenate([np.random.choice(event_times[(event_times > flare_begin[i]) & (event_times < flare_end[i])], size = final_events_per_flare, replace=False) for i in range(len(flare_begin))])
    accepted_time_indices = unsorted_search(event_times, accepted_event_times)

    accepted_event_ra, accepted_event_dec, accepted_event_theta, accepted_event_lst = event_ra[accepted_time_indices], event_dec[accepted_time_indices], event_theta[accepted_time_indices], event_lst[accepted_time_indices]
    accepted_flare_ra = np.concatenate(np.transpose(np.array([flare_ra for i in range(final_events_per_flare)])))
    accepted_flare_dec = np.concatenate(np.transpose(np.array([flare_dec for i in range(final_events_per_flare)])))

    #accepted_flare_dec =
    #save flare events in dataframe
    flare_events = pd.DataFrame(zip(accepted_event_times, np.degrees(accepted_event_ra), np.degrees(accepted_event_dec), np.degrees(accepted_event_theta), np.degrees(accepted_event_lst)), columns=['gps_time', 'ra', 'dec', 'theta', 'lst'])

    flare_events['is_from_flare'] = pd.Series(np.full(len(flare_events.index), True, dtype=bool))
    flare_events['flare_dec'] = pd.Series(np.degrees(accepted_flare_dec))
    flare_events['flare_ra'] = pd.Series(np.degrees(accepted_flare_ra))

    return flare_events

#load events from isotropic distribution
#define the number of flares, events per flare and duration of flare in seconds
if (len(sys.argv) < 5):
    print('You must give file to contaminate, number of flares, number of evenents per flare and duration of flare in days')
    exit()

#save file name
filename = sys.argv[1]

#checks if file exists
if not os.path.exists(filename):
    print("Requested file does not exist. Aborting")
    exit()

#save path to file and its basename
path_name = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]

#save dataframe with isotropic events
event_data = pd.read_parquet(filename, engine = 'fastparquet')

#flag events that are not from explosion
event_data['is_from_flare'] = pd.Series(np.full(len(event_data.index), False, dtype=bool))
event_data['flare_dec'] = pd.Series(np.full(len(event_data.index), np.nan))
event_data['flare_ra'] = pd.Series(np.full(len(event_data.index), np.nan))

print(event_data.head(10))

#save dataframe with flares
flare_catalog = './datasets/MockFlares_1000_2010-01-01_2020-01-01.parquet'
flare_data = pd.read_parquet(flare_catalog, engine='fastparquet')

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#define the maximum value of theta
theta_max = np.radians(80)

#filter flares outside field of view of observatory
flare_data = flare_data[flare_data['dec_flare'] < np.degrees(theta_max + lat_pao)]

n_flares = int(sys.argv[2])
flare_duration_days = int(sys.argv[3])
flare_duration = int(sys.argv[3])*86_164
final_events_per_flare = int(sys.argv[4])
initial_events_per_flare = 100*final_events_per_flare

#define the angular resolution
ang_resol = np.radians(1)

#fix seed for choosing a sample of flares from dataframe
seed_flare = 10

#sample n_flares from flare catalog
start_time = datetime.now()

flare_data = flare_data.sample(n_flares, random_state=seed_flare).reset_index()

print(flare_data)

#producing flare events
flare_events = get_flare_events(flare_data, flare_duration, initial_events_per_flare, final_events_per_flare, ang_resol, seed_flare, pao_loc, theta_max)

print(flare_events)

#substituting events in ud distribution with flare events
selected_events = event_data.sample(len(flare_events.index), random_state=seed_flare)
event_data = event_data.drop(selected_events.index)
event_data.reset_index(drop=True, inplace=True)
event_data = event_data.append(flare_events, ignore_index=True)

#order events by time
event_data.sort_values('gps_time', inplace = True, ignore_index=True)

print(event_data)

print('Producing isotropic sample with flare events took', datetime.now() - start_time,'s')

output_path = './datasets/events_with_flares/'

#create directory whose name includes number of flares and events per flare
output_dir = 'nFlares_%i_nEventsPerFlare_%i_FlareDuration_%i' % (n_flares, final_events_per_flare, flare_duration_days)
output_path = output_path + output_dir

if not os.path.exists(output_path):
    os.makedirs(output_path)

#save scrambled events
event_data.to_parquet(os.path.join(output_path, basename + '_nFlare_%i_nEventsPerFlare_%i_FlareDuration_%i.parquet' % (n_flares, final_events_per_flare, flare_duration_days)), index=True)
