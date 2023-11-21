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

#to confirm that events are being correctly sampled
import matplotlib.pyplot as plt

def ra_to_phi(ra):
    return np.where(ra > np.pi, 2*np.pi - ra, -ra)

#compute phi for u uniform in [0, 1]
def compute_phi(u):
    return 2*np.pi*u

#convert zenith to altitude
def zenith_to_alt(theta):
    return np.pi/2 - theta

#scramble events by suffling event sidereal time and theta independently, and sampling phi from uniform dist
def scramble_events(event_data, pao_loc):

    #save events as arrays to shuffle
    event_time = Time(event_data['gps_time'].to_numpy(), format='gps', scale='utc', location=pao_loc).gps
    event_theta = np.radians(event_data['theta'].to_numpy())
    event_phi = np.radians(event_data['phi'].to_numpy())

    #delete dataframe to save memory
    del event_data

    #shuffle event time
    np.random.shuffle(event_time)
    event_time = Time(event_time, format='gps', scale='utc', location=pao_loc)

    #shuffle theta
    np.random.shuffle(event_theta)

    #shuffle phi
    np.random.shuffle(event_phi)

    #compute the corresponding right ascensions and declinations
    horizontal_coords = SkyCoord(az=event_phi*u.rad, alt=zenith_to_alt(event_theta)*u.rad, frame=AltAz(obstime=event_time, location=pao_loc))
    equatorial_coords = horizontal_coords.transform_to('icrs')

    event_ra = equatorial_coords.ra.rad
    event_dec = equatorial_coords.dec.rad

    #compute sidereal time for each event
    event_lst = event_time.sidereal_time('apparent').rad

    #order events by time
    event_time, event_ra, event_dec, event_theta, event_phi, event_lst = time_ordered_events(event_time.gps, event_ra, event_dec, event_theta, event_phi, event_lst)

    #save result as dataframe
    event_data = pd.DataFrame(zip(event_time, np.degrees(event_ra), np.degrees(event_dec), np.degrees(event_theta), np.degrees(event_phi), np.degrees(event_lst)), columns = ['gps_time', 'ra', 'dec', 'theta', 'phi', 'lst'])

    return event_data

#set the seed of numpy's random number generator
seed = 47
np.random.seed(seed)

#define the right ascencion and declination of the center of the patch
dec_source = 0
ra_source = 0

#fetch the file with the isotropic sample that contains the requested coordinates for the center of the patch
input_path = './datasets'

iso_filename = [os.path.join(input_path, file) for file in os.listdir(input_path) if 'decCenter_%i_raCenter_%i' % (dec_source, ra_source) in file]


if len(iso_filename) != 1:
    print('Too many files found!')
    exit()

#save file name
filename = iso_filename[0]

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

#plot skymap of events
fig = plt.figure()
ax = fig.add_subplot(111, projection='hammer')

# plot skymaps of penalized pvalues
ax.scatter(ra_to_phi(np.radians(event_data['ra'])), np.radians(event_data['dec']), marker='o', color='tab:blue')
#ax_skymap_poisson_pvalue.grid()

plt.show()

#scrambling events
start_time = datetime.now()

for i in range(5):

    event_data = scramble_events(event_data, pao_loc)

print('Scrambling events took', datetime.now() - start_time,'s')

event_data.hist(column='theta')
plt.show()





#save scrambled events
output_path = './scrambled_events/decCenter_%i_raCenter_%i' % (dec_source, ra_source)

#event_data.to_parquet(path_name + '/scrambled_events/Scrambled_' + basename + '_%i.parquet' % sample_index, index=True)
