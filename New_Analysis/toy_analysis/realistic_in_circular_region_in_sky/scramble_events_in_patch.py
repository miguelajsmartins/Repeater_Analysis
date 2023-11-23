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
from array_manip import unsorted_search

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

#convert zenith to altitude
def alt_to_zenith(theta):
    return np.pi/2 - theta

#compute the minimum and maximum azimuth for an event with a given theta, assuming that theta < theta_max
def compute_azimuth_limits(psi_patch, theta, theta_center, phi_center):

    #compute max and min phi under different conditions
    event_closer_to_zenith = theta <= psi_patch - theta_center
    event_closer_to_center = np.logical_not(event_closer_to_zenith)

    delta_phi = np.ones(len(theta_center))
    phi_min = np.ones(len(theta_center))
    phi_max = np.ones(len(theta_center))

    #center of patch at zenith
    phi_min[event_closer_to_zenith] = 0
    phi_max[event_closer_to_zenith] = 2*np.pi

    delta_phi[event_closer_to_center] = np.arccos( (np.cos(psi_patch) - np.cos(theta_center[event_closer_to_center])*np.cos(theta[event_closer_to_center])) / (np.sin(theta_center[event_closer_to_center])*np.sin(theta[event_closer_to_center])) )

    phi_min[event_closer_to_center] = phi_center[event_closer_to_center] - delta_phi[event_closer_to_center]
    phi_max[event_closer_to_center] = phi_center[event_closer_to_center] + delta_phi[event_closer_to_center]

    plt.close()
    plt.plot(np.degrees(theta), (np.cos(psi_patch) - np.cos(theta_center[event_closer_to_center])*np.cos(theta[event_closer_to_center])) / (np.sin(theta_center[event_closer_to_center])*np.sin(theta[event_closer_to_center])), linestyle = 'None', marker = 'o')
    plt.show()

    return phi_min, phi_max

#scramble events by suffling event sidereal time and theta independently, and sampling phi from uniform dist, within allowed limits
def scramble_events(event_data, psi_patch, dec_center, ra_center, pao_loc):

    #save events as arrays to shuffle
    event_time = Time(event_data['gps_time'].to_numpy(), format='gps', scale='utc', location=pao_loc).gps
    event_theta = np.radians(event_data['theta'].to_numpy())
    event_phi = np.radians(event_data['phi'].to_numpy())

    n_events = len(event_time)

    #event_phi_min = np.radians(event_data['phi_min'].to_numpy())
    #event_phi_max = np.radians(event_data['phi_max'].to_numpy())

    #delete dataframe to save memory
    del event_data

    #shuffle event time
    np.random.shuffle(event_time)

    event_time = Time(event_time, format='gps', scale='utc', location=pao_loc)

    #compute sidereal time for each event
    event_lst = event_time.sidereal_time('apparent').rad

    #compute the position of the center of the patch
    center_horizontal_coordinates = SkyCoord(ra_center*u.rad, dec_center*u.rad, frame='icrs').transform_to(AltAz(obstime=event_time, location=pao_loc))
    theta_center = alt_to_zenith(center_horizontal_coordinates.alt.rad)
    phi_center = center_horizontal_coordinates.az.rad

    #verifies which theta values are in the patch, and shuffles those events
    new_event_theta = []

    while len(new_event_theta) < n_events:

        #compute 
        patch_event_ang_diff = ang_diff(np.pi/ 2 - theta, np.pi / 2 - theta_center, phi_center, event_phi)

    is_in_patch = ang_diff < psi_patch

    event_theta_in_patch = event_theta[is_in_patch]

    phi_min, phi_max = compute_azimuth_limits(psi_patch, event_theta, theta_center, phi_center)

    #random sample phi within the limits
    event_phi = (phi_max - phi_min) * np.random.random(len(event_theta)) + phi_min

    #compute the corresponding right ascensions and declinations
    horizontal_coords = SkyCoord(az=event_phi*u.rad, alt=zenith_to_alt(event_theta)*u.rad, frame=AltAz(obstime=event_time, location=pao_loc))
    equatorial_coords = horizontal_coords.transform_to('icrs')

    event_ra = equatorial_coords.ra.rad
    event_dec = equatorial_coords.dec.rad

    #order events by time
    event_time, event_gps_time, event_ra, event_dec, event_theta, event_phi, event_lst = time_ordered_events(event_time, event_ra, event_dec, event_theta, event_phi, event_lst)

    #save result as dataframe
    event_data = pd.DataFrame(zip(event_gps_time, np.degrees(event_ra), np.degrees(event_dec), np.degrees(event_theta), np.degrees(event_phi), np.degrees(event_lst)), columns = ['gps_time', 'ra', 'dec', 'theta', 'phi', 'lst'])

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

print(min(event_data['phi_max'] - event_data['phi_min']))

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#save the radius of the patch
patch_radius = np.radians(float(basename.split('_')[7]))

#scrambling events
start_time = datetime.now()

new_event_data = scramble_events(event_data, patch_radius, dec_source, ra_source, pao_loc)

print('Scrambling events took', datetime.now() - start_time,'s')

#plot skymap of events
fig = plt.figure()
ax = fig.add_subplot(111, projection='hammer')

# plot skymaps of penalized pvalues
ax.scatter(ra_to_phi(np.radians(new_event_data['ra'])), np.radians(new_event_data['dec']), marker='o', s = .1, color='tab:blue')
#ax_skymap_poisson_pvalue.grid()

plt.show()

plt.close()

event_data.hist(column='theta')
plt.show()


# #save scrambled events
# output_path = './scrambled_events/decCenter_%i_raCenter_%i' % (dec_source, ra_source)

#event_data.to_parquet(path_name + '/scrambled_events/Scrambled_' + basename + '_%i.parquet' % sample_index, index=True)
