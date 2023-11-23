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

import matplotlib.pyplot as plt

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import compute_directional_exposure

#convert zenith to altitude
def zenith_to_alt(theta):
    return np.pi/2 - theta

#convert zenith to altitude
def alt_to_zenith(theta):
    return np.pi/2 - theta

#compute theta assuming that the directional exposure depends only on cosine theta
def compute_theta(u, theta_min, theta_max):

    return .5*np.arccos(np.cos(2*theta_min) - u * (np.cos(2*theta_min) - np.cos(2*theta_max)) )

def compute_phi(u, phi_min, phi_max):

    return (phi_max - phi_min)*u + phi_min

#compute declination for u uniform in [0, 1]
def compute_dec(u, dec_max, dec_min):

    return np.arcsin(u*(np.sin(dec_max) - np.sin(dec_min)) + np.sin(dec_min))

#compute right ascension for u uniform in [0, 1]
def compute_ra(u, dec_center, ra_center, psi_patch, dec):

    #compute the maximum and minimum right ascensions
    delta_ra = np.arccos( (np.cos(psi_patch) - np.sin(dec_center)*np.sin(dec)) / (np.cos(dec)*np.cos(dec_center)) )
    ra_min = ra_center - delta_ra
    ra_max = ra_center + delta_ra

    return (ra_max - ra_min)*u + ra_min

#compute right ascention limits for a given declination and a circular patch
def get_max_ra_width(dec, dec_min, dec_max, dec_center, patch_radius):

    ra_width = np.ones(len(dec))

    #computes max ra width for different cases
    dec_extreme = np.logical_or(dec == dec_max, dec == dec_min)
    dec_not_extreme = np.logical_not(dec_extreme)

    ra_width[dec_extreme] = 0
    ra_width[dec_not_extreme] = np.arccos(( np.cos(patch_radius) - np.sin(dec[dec_not_extreme])*np.sin(dec_center) ) / (np.cos(dec_center)*np.cos(dec[dec_not_extreme])))

    return 2*ra_width

#compute the theta limits given the position of the center of the patch
def compute_zenith_limits(psi_patch, time, ra_center, dec_center, pao_loc, theta_min, theta_max):

    #compute coordinates of patch in the observatory reference frame
    patch_coordinates = SkyCoord(ra_center*u.rad, dec_center*u.rad, frame='icrs').transform_to(AltAz(obstime=time, location=pao_loc))

    theta_center = alt_to_zenith(patch_coordinates.alt.rad)
    phi_center = patch_coordinates.az.rad

    #compute theta limits
    min = np.maximum(theta_min, theta_center - psi_patch)
    max = np.minimum(theta_max, theta_center + psi_patch)

    #filters events outside field of view
    outside_fov = min > theta_max
    #inside_fov = np.logical_not(outside_fov)

    min[outside_fov] = np.nan
    max[outside_fov] = np.nan

    return theta_center, phi_center, min, max

#compute the minimum and maximum azimuth for an event with a given theta, assuming that theta < theta_max
def compute_azimuth_limits(psi_patch, theta, theta_center, phi_center):

    #compute max and min phi under different conditions
    event_closer_to_zenith = theta <= psi_patch - theta_center
    event_closer_to_center = np.logical_not(event_closer_to_zenith)

    delta_phi = np.ones(theta_center.shape)
    phi_min = np.ones(theta_center.shape)
    phi_max = np.ones(theta_center.shape)

    #center of patch at zenith
    phi_min[event_closer_to_zenith] = 0
    phi_max[event_closer_to_zenith] = 2*np.pi

    delta_phi[event_closer_to_center] = np.arccos( (np.cos(psi_patch) - np.cos(theta_center[event_closer_to_center])*np.cos(theta[event_closer_to_center])) / (np.sin(theta_center[event_closer_to_center])*np.sin(theta[event_closer_to_center])) )

    phi_min[event_closer_to_center] = phi_center[event_closer_to_center] - delta_phi[event_closer_to_center]
    phi_max[event_closer_to_center] = phi_center[event_closer_to_center] + delta_phi[event_closer_to_center]

    return phi_min, phi_max

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

    return time[accepted_indices], ra[accepted_indices], dec[accepted_indices], theta[accepted_indices], azimuth, lst[accepted_indices]

#compute accepted events using directly the generated theta
def get_equatorial_coordinates(time, theta, phi, pao_loc):

    #compute equatorial coordinates
    horizontal_coords = SkyCoord(az=phi*u.rad, alt=zenith_to_alt(theta)*u.rad, frame=AltAz(obstime=time, location=pao_loc))
    equatorial_coords = horizontal_coords.transform_to('icrs')

    ra = equatorial_coords.ra.rad
    dec = equatorial_coords.dec.rad

    #compute sidereal time for each event
    lst = time.sidereal_time('apparent').rad

    return ra, dec, lst


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
patch_radius = np.radians(15)

#define the position of a source
dec_center = 0
ra_center = 0

#define the number of events
n_events = 100_000

#define the maximum accepted zenith angle
theta_min = np.radians(0)
theta_max = np.radians(80)
all_dec_max = lat_pao + theta_max

dec_min = dec_center - patch_radius
dec_max = dec_center + patch_radius

if dec_max > all_dec_max:
    print('please make sure all events in patch are in FoV of observatory!')
    exit()

#compute the integrated exposure over the circular area to determine the number of expected events in the cap
dec_in_patch = np.linspace(dec_min, dec_max, 1000)
ra_width_in_patch = get_max_ra_width(dec_in_patch, dec_min, dec_max, dec_center, patch_radius)

dec_range = np.linspace(-np.pi / 2, all_dec_max, 10000)

exposure_in_patch = compute_directional_exposure(dec_in_patch, theta_max, lat_pao)
total_exposure = compute_directional_exposure(dec_range, theta_max, lat_pao)

integrated_exposure_in_patch = np.trapz(exposure_in_patch*np.cos(dec_in_patch)*ra_width_in_patch, x = dec_in_patch)
total_integrated_exposure = 2*np.pi*np.trapz(total_exposure*np.cos(dec_range), x = dec_range)

n_events_accepted_in_patch = np.ceil((integrated_exposure_in_patch / total_integrated_exposure)*n_events).astype('int')

#multiply the computed number of events by factor that takes into account the rejection of events
print(n_events_accepted_in_patch)

n_events_in_patch = int(2*n_events_accepted_in_patch)

#start of program
start_time = datetime.now()

n_samples = 20
#for i in range(10):

#generate a collection of time stamps
time = np.random.randint(start_date, end_date, size = (n_events_in_patch, n_samples))
time = Time(time, format='gps', scale='utc', location=pao_loc)

#generate vectors of uniformly distributed variables
rand_a = np.random.random((n_events_in_patch, n_samples))
rand_b = np.random.random((n_events_in_patch, n_samples))

start_time_1 = datetime.now()

print('Generating random numbers took', start_time_1 - start_time)

#given the position of the source compute the limits for theta
theta_center, phi_center, theta_lower, theta_upper = compute_zenith_limits(patch_radius, time, ra_center, dec_center, pao_loc, theta_min, theta_max)

#compute theta given the provided limits
theta = compute_theta(rand_a, theta_lower, theta_upper)

start_time_2 = datetime.now()

print('Computing theta took', start_time_2 - start_time_1)

#compute phi limits given theta
phi_min, phi_max = compute_azimuth_limits(patch_radius, theta, theta_center, phi_center)

#compute corresponding values of phi
phi = compute_phi(rand_b, phi_min, phi_max)

start_time_3 = datetime.now()

print('Computing phi took', start_time_3 - start_time_2)

#cleans arrays
theta_not_nan = np.logical_not(np.isnan(theta))

time, theta, phi = time[theta_not_nan], theta[theta_not_nan], phi[theta_not_nan]

ra, dec, lst = get_equatorial_coordinates(time, theta, phi, pao_loc)

start_time_4 = datetime.now()

print('Computing equatorial coordinates took', start_time_4 - start_time_3)

#compute theta and phi for each value of u and v
#theta = compute_dec(rand_a, dec_min, dec_max)
#phi = compute_ra(rand_b, dec_center, ra_center, patch_radius, dec)

#accept events and save first few accepted events in file


#accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = compute_accepted_events(time, ra, dec, pao_loc, theta_max)

print('Efficiency in accepting events =', time.shape[0] / n_events_in_patch)
print('Entire analsyis took ', datetime.now() - start_time, ' s')

#accept only the first n_events_accepted_in_patch events
# accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = accepted_time[:n_events_accepted_in_patch], accepted_ra[:n_events_accepted_in_patch], accepted_dec[:n_events_accepted_in_patch], accepted_theta[:n_events_accepted_in_patch], accepted_phi[:n_events_accepted_in_patch], accepted_lst[:n_events_accepted_in_patch]
#
# #order events by time
# accepted_time, accepted_gps_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_phi, accepted_lst)
#
# #compute the azimuth limits for a given theta
# theta_source = ang_diff(dec_center, lat_pao, ra_center, accepted_lst)
# phi_source = SkyCoord(accepted_ra*u.rad, accepted_dec*u.rad, frame='icrs').transform_to(AltAz(obstime=accepted_time, location=pao_loc)).az.rad
# accepted_phi_min, accepted_phi_max = compute_azimuth_limits(patch_radius, accepted_theta, theta_source, phi_source)
#
# #save accepted events into a dataframe
# accepted_event_data = pd.DataFrame(zip(accepted_gps_time, np.degrees(accepted_ra), np.degrees(accepted_dec), np.degrees(accepted_theta), np.degrees(accepted_phi), np.degrees(accepted_lst)), columns=['gps_time', 'ra', 'dec', 'theta', 'phi', 'lst'])
#
# print(accepted_event_data)
#
# #define the output path
# output_path = './datasets'
#
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#
# accepted_event_data.to_parquet('./datasets/IsoDist_%i_decCenter_%i_raCenter_%i_patchRadius_%.0f_acceptance_th80_10years.parquet' % (int(n_events_accepted_in_patch), dec_center, ra_center, np.degrees(patch_radius)), index=True)
