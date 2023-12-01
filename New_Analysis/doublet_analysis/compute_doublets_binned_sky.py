import numpy as np
import pandas as pd
import healpy as hp

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys
import os

sys.path.append('./src/')

import time

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map
from event_manip import local_LiMa_significance
from array_manip import unsorted_search

import matplotlib.pyplot as plt

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#converts declination into colat
def dec_to_colat(dec):
    return np.pi / 2 - dec

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE, bin_indexes):

    #get colatidude and ra of each bin center in healpy sky
    colat, ra = hp.pix2ang(NSIDE, bin_indexes)

    #convert to declination
    #dec = colat_to_dec(colat)

    return ra, colat

#computes the number of doublets in each target
def compute_doublets_per_target(event_data, NSIDE, tau):

    #get the bin indices of cosmic ray events
    event_time = event_data['gps_time'].to_numpy()
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_ra = np.radians(event_data['ra'].to_numpy())

    event_indices = hp.ang2pix(NSIDE, dec_to_colat(event_dec), event_ra)

    #order indices and event indices accordingly
    sorted_indices = event_indices.argsort()

    event_indices = event_indices[sorted_indices]
    event_time, event_dec, event_ra = event_time[sorted_indices], event_dec[sorted_indices], event_ra[sorted_indices]

    #define a vector with unique indices
    unique_indices, counts = np.unique(event_indices, return_counts = True)

    #compute the splitting points
    splitting_points = np.cumsum(counts)[:-1]

    #group events per target
    event_time_per_target = np.split(event_time, splitting_points)
    event_dec_per_target = np.split(event_dec, splitting_points)
    event_ra_per_target = np.split(event_ra, splitting_points)

    #cleans targets where the number of events is only 1
    event_time_per_target = [array for array in event_time_per_target if len(array) > 1]
    event_dec_per_target = [array for array in event_dec_per_target if len(array) > 1]
    event_ra_per_target = [array for array in event_ra_per_target if len(array) > 1]

    #complete arrays for manipulation as a matrix
    max_size = max([len(array) for array in event_time_per_target])

    event_time_per_target = np.array([np.append(array, np.full(max_size - len(array), np.nan)) for array in event_time_per_target])
    event_dec_per_target = np.array([np.append(array, np.full(max_size - len(array), np.nan)) for array in event_dec_per_target])
    event_ra_per_target = np.array([np.append(array, np.full(max_size - len(array), np.nan)) for array in event_ra_per_target])

    #order events in time within each bin
    time_ordered_indices = event_time_per_target.argsort(axis = 1)

    event_time_per_target = np.take_along_axis(event_time_per_target, time_ordered_indices, axis = 1)
    event_dec_per_target, event_ra_per_target = np.take_along_axis(event_dec_per_target, time_ordered_indices, axis = 1), np.take_along_axis(event_ra_per_target, time_ordered_indices, axis = 1)

    #compute the time differences between events
    time_diff = np.diff(event_time_per_target, axis = 1)

    #selects doublets separated by less than tau
    is_doublet = time_diff < tau

    doublets = np.column_stack([
        event_time_per_target[:, :-1][is_doublet].ravel(),
        event_time_per_target[:, 1:][is_doublet].ravel(),
        event_ra_per_target[:, :-1][is_doublet].ravel(),
        event_ra_per_target[:, 1:][is_doublet].ravel(),
        event_dec_per_target[:, :-1][is_doublet].ravel(),
        event_dec_per_target[:, 1:][is_doublet].ravel()
    ])

    #save the doublets in dataframe
    column_names = ['gps_time_1', 'gps_time_2', 'ra_1', 'ra_2', 'dec_1', 'dec_2']

    doublet_data = pd.DataFrame(doublets, columns = column_names)

    return doublet_data


#define the input path and save files with event samples
input_path = './datasets/iso_samples'
input_filelist = []

for file in os.listdir(input_path):

    filename = os.path.join(input_path, file)

    if os.path.isfile(filename):

        input_filelist.append(filename)

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#defines the maximum declination
theta_max = np.radians(80)
dec_max = pao_lat + theta_max

#defines the NSIDE parameter
NSIDE = 32
npix = hp.nside2npix(NSIDE)
target_area = (4*np.pi / npix)

#define the tau parameter
tau = 86_164 #in seconds

#compute the coordinates of the center of each bin and transform them into ra and dec. Compute the corresponding exposure map
all_pixel_indices = np.arange(npix)
ra_target, colat_center = get_binned_sky_centers(NSIDE, all_pixel_indices)
dec_target = colat_to_dec(colat_center)

#filter out pixels outside the FoV of the observatory
inside_fov = dec_target < dec_max
ra_target, dec_target = ra_target[inside_fov], dec_target[inside_fov]

#order events by declination
sorted_indices = dec_target.argsort()
ra_target, dec_target = ra_target[sorted_indices], dec_target[sorted_indices]

#compute the normalized exposure
unique_dec_target = np.unique(dec_target)
unique_exposure = compute_directional_exposure(unique_dec_target, theta_max, pao_lat)
integrated_exposure = 2*np.pi*np.trapz(unique_exposure*np.cos(unique_dec_target), x = unique_dec_target)

#compute the normalized exposure map
exposure_map = compute_directional_exposure(dec_target, theta_max, pao_lat) / integrated_exposure

#define the output path
output_path = './datasets/iso_doublets'

start = datetime.now()

#compute the number of doublets for each sample
for input_file in input_filelist:

    #save path to file and its basename
    basename = os.path.splitext(os.path.basename(input_file))[0]

    #save dataframe with isotropic events
    event_data = pd.read_parquet(input_file, engine = 'fastparquet')

    #compute number of events and observation time
    n_events = len(event_data.index)
    obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]

    #compute the expected number of events in each target
    mu_per_target = n_events*exposure_map*target_area
    rate_per_target = mu_per_target

    doublet_data = compute_doublets_per_target(event_data, NSIDE, tau)

    #clean nan and duplicated events
    doublet_data.dropna(inplace = True, ignore_index = True)
    doublet_data.drop_duplicates(inplace = True, ignore_index = True)

    #convert angles to degrees
    doublet_data[['ra_1', 'ra_2', 'dec_1', 'dec_2']] = np.degrees(doublet_data[['ra_1', 'ra_2', 'dec_1', 'dec_2']])

    print(doublet_data)

    #save doublet data into a parquet file
    output_file = os.path.join(output_path, 'Doublets_binnedSky_' + basename)
    doublet_data.to_parquet(output_file, index = True)

print('This took', datetime.now() - start, 's')


# #defines the time window
# time_window = [86_164, 7*86_164, 30*86_164] #consider a time window of a single day
#
# #compute event average event rate in angular window
# target_radius = np.radians(1)
# rate = (n_events / obs_time)
#
# #computing estimators for a given skymap of events
# start_time = datetime.now()
#
# estimator_data = compute_estimators(event_data, NSIDE, rate, target_radius, time_window, theta_max, pao_lat)
#
# end_time = datetime.now() - start_time
#
# print('Computing estimators took', end_time,'s')
#
# #order events by declination
# estimator_data = estimator_data.sort_values('dec_target')
# estimator_data = estimator_data.reset_index(drop=True)
#
# #prints table with summary of data
# print(estimator_data)
# print(estimator_data.describe())
#
# #saves estimator data
# estimator_data.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
