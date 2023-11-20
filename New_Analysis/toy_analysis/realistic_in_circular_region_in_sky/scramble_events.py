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

#compute phi for u uniform in [0, 1]
def compute_phi(u):
    return 2*np.pi*u

#convert zenith to altitude
def zenith_to_alt(theta):
    return np.pi/2 - theta

#scramble events by suffling event sidereal time and theta independently, and sampling phi from uniform dist
def scramble_events(event_data, pao_loc):

    #set seed
    #seed(0)

    #save events as arrays to shuffle
    event_time = Time(event_data['gps_time'].to_numpy(), format='gps', scale='utc', location=pao_loc).gps
    event_theta = np.radians(event_data['theta'].to_numpy())

    #delete dataframe to save memory
    del event_data

    #shuffle event time
    np.random.shuffle(event_time)
    event_time = Time(event_time, format='gps', scale='utc', location=pao_loc)

    #shuffle theta
    np.random.shuffle(event_theta)

    #produce random phi
    event_phi = compute_phi(np.random.random(len(event_time)))

    #compute the corresponding right ascensions and declinations
    horizontal_coords = SkyCoord(az=event_phi*u.rad, alt=zenith_to_alt(event_theta)*u.rad, frame=AltAz(obstime=event_time, location=pao_loc))
    equatorial_coords = horizontal_coords.transform_to('icrs')

    event_ra = equatorial_coords.ra.rad
    event_dec = equatorial_coords.dec.rad

    #compute sidereal time for each event
    event_lst = event_time.sidereal_time('apparent').rad

    #order events by time
    event_time, event_ra, event_dec, event_theta, event_lst = time_ordered_events(event_time.gps, event_ra, event_dec, event_theta, event_lst)

    #save result as dataframe
    event_data = pd.DataFrame(zip(event_time, np.degrees(event_ra), np.degrees(event_dec), np.degrees(event_theta), np.degrees(event_lst)), columns = ['gps_time', 'ra', 'dec', 'theta', 'lst'])

    return event_data

#load events from isotropic distribution
if (len(sys.argv) < 3):
    print('Must give a file containing isotropic distribution of events and index of file')
    exit()

#save file name
filename = sys.argv[1]
sample_index = int(sys.argv[2])

#checks if file exists
if not os.path.exists(filename):
    print("Requested file does not exist. Aborting")
    exit()

#save path to file and its basename
path_name = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]

#save dataframe with isotropic events
event_data = pd.read_parquet(filename, engine = 'fastparquet')

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

start_time = datetime.now()

#scrambling events
seed = seed(open('/dev/random','rb').read(4))

event_data = scramble_events(event_data, pao_loc)

print('Scrambling events took', datetime.now() - start_time,'s')

#save scrambled events
event_data.to_parquet(path_name + '/scrambled_events/Scrambled_' + basename + '_%i.parquet' % sample_index, index=True)
