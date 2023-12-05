import numpy as np
import pandas as pd
import healpy as hp

import numpy.ma as ma

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys
import os

sys.path.append('../src/')

import time

from event_manip import ang_diff
from event_manip import compute_directional_exposure
from event_manip import compute_lambda_correction

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

    return ra, colat

#filter out bins such that each target is partially or totally outside the patch
def get_bins_in_patch(ra_target, dec_target, ra_center, dec_center, patch_radius, target_radius):

    #define the minimum and maximum declinations
    dec_min = dec_center - patch_radius + target_radius
    dec_max = dec_center + patch_radius - target_radius

    #keeps only targets in the defined declination band
    in_band = np.logical_and(dec_target > dec_min, dec_target < dec_max)

    ra_target_in_band, dec_target_in_band = ra_target[in_band], dec_target[in_band]

    #for reach declination in the band compute the minimum and maximum right ascension values
    arg_delta_ra = (np.cos(patch_radius - target_radius) - np.sin(dec_target_in_band)*np.sin(dec_center)) / (np.cos(dec_center)*np.cos(dec_target_in_band))
    delta_ra = np.arccos(arg_delta_ra)

    ra_1 = ra_center + delta_ra
    ra_2 = 2*np.pi + ra_center - delta_ra

    ra_right = np.minimum(ra_1,ra_2)
    ra_left = np.maximum(ra_1,ra_2)

    #saves only the binds inside patch
    in_patch = np.logical_or(ra_target_in_band < ra_right, ra_target_in_band > ra_left)

    ra_target_in_patch, dec_target_in_patch = ra_target_in_band[in_patch], dec_target_in_band[in_patch]

    return ra_target_in_patch, dec_target_in_patch

def compute_estimators_per_target(event_data, ra_target, dec_target, target_radius, exposure_map, obs_time, n_events):

    #saves events
    event_time = event_data['gps_time'].to_numpy()
    event_dec = np.radians(event_data['dec'].to_numpy())
    event_ra = np.radians(event_data['ra'].to_numpy())

    #sorts events by declination to improve computation efficiency
    sorted_indices = event_dec.argsort()
    event_time, event_dec, event_ra = event_time[sorted_indices], event_dec[sorted_indices], event_ra[sorted_indices]

    #split sky in bands of constant declination
    unique_dec_target, counts_dec_target = np.unique(dec_target, return_counts=True)
    split_points = np.cumsum(counts_dec_target)[:-1]
    splitted_dec_target = np.split(dec_target, split_points)
    splitted_ra_target = np.split(ra_target, split_points)

    #compute the expected number of events per target and the expected rate
    target_area = 2*np.pi*(1 - np.cos(target_radius))
    mu_per_target = target_area*exposure_map[np.searchsorted(dec_target, unique_dec_target)]*n_events
    rate_per_target = mu_per_target / obs_time

    #to save the values of the estimators and other important data
    n_events_per_target_list = []
    lambda_per_target_list = []
    lambda_corrected_per_target_list = []

    #doublet_per_target = []

    for i, (splitted_dec_t, ra_t) in enumerate(zip(splitted_dec_target, splitted_ra_target)):

        #filters events outside declination band
        in_dec_band = np.abs(event_dec - splitted_dec_t[0]) < target_radius

        event_time_in_band, event_dec_in_band, event_ra_in_band = event_time[in_dec_band], event_dec[in_dec_band], event_ra[in_dec_band]

        #to allow numpy manipulation
        dummy_grid, event_time_grid = np.meshgrid(splitted_dec_t, event_time_in_band)
        dec_t_grid, event_dec_grid = np.meshgrid(splitted_dec_t, event_dec_in_band)
        ra_t_grid, event_ra_grid = np.meshgrid(ra_t, event_ra_in_band)

        #compute the angular separation between events and target
        ang_sep = ang_diff(event_dec_grid, dec_t_grid, event_ra_grid, ra_t_grid)

        #groups events per target
        in_target = ang_sep < target_radius

        event_time_in_target = ma.masked_array(event_time_grid, mask=np.logical_not(in_target)).filled(fill_value=np.nan).T
        event_dec_in_target = ma.masked_array(event_dec_grid, mask=np.logical_not(in_target)).filled(fill_value=np.nan).T
        event_ra_in_target = ma.masked_array(event_ra_grid, mask=np.logical_not(in_target)).filled(fill_value=np.nan).T

        #sorts events by time
        sorted_time_indices = event_time_in_target.argsort(axis=1)

        event_time_in_target = np.take_along_axis(event_time_in_target, sorted_time_indices, axis=1)
        event_dec_in_target = np.take_along_axis(event_dec_in_target, sorted_time_indices, axis=1)
        event_ra_in_target = np.take_along_axis(event_ra_in_target, sorted_time_indices, axis=1)

        #computes the number of events per target
        n_events_per_target = np.array([ len(array[np.logical_not(np.isnan(array))]) for array in event_time_in_target])

        #computes the time difference between consecutive events
        time_diff = np.diff(event_time_in_target, axis=1)
        time_diff = time_diff*rate_per_target[i]

        #computes the regular lambda
        lambda_value = - np.nansum(np.log(time_diff), axis = 1)

        #computes the corrected value of lambda
        lambda_corrected_value = lambda_value - compute_lambda_correction(n_events_per_target, mu_per_target[i])

        #saves important estimators
        n_events_per_target_list.append(n_events_per_target)
        lambda_per_target_list.append(lambda_value)
        lambda_corrected_per_target_list.append(lambda_corrected_value)

        #print(splitted_dec_t)
        #print(mu_per_target[i])

        # is_doublet = time_diff < tau
        #
        # #forms event doublets
        # doublets = np.column_stack([
        #     event_time_in_target[:, :-1][is_doublet].ravel(),
        #     event_time_in_target[:, 1:][is_doublet].ravel(),
        #     event_ra_in_target[:, :-1][is_doublet].ravel(),
        #     event_ra_in_target[:, 1:][is_doublet].ravel(),
        #     event_dec_in_target[:, :-1][is_doublet].ravel(),
        #     event_dec_in_target[:, 1:][is_doublet].ravel()
        # ])
        #
        # doublet_per_target.append(doublets)

        #if i % 10 == 0:
            #print('%i / %i target bands done!' % (i, len(unique_dec_target)))

    #doublet_per_target = np.concatenate(doublet_per_target, axis=0)

    #convert lists to arrays
    n_events_per_target_list = np.concatenate(n_events_per_target_list, axis = 0)
    lambda_per_target_list = np.concatenate(lambda_per_target_list, axis = 0)
    lambda_corrected_per_target_list = np.concatenate(lambda_corrected_per_target_list, axis = 0)

    #get all mu values
    full_mu_per_target = target_area*exposure_map*n_events

    #save computed estimators
    column_names = ['ra_target', 'dec_target', 'mu_in_target', 'n_events_in_target', 'lambda', 'lambda_corrected']
    estimator_data = zip(np.degrees(ra_target), np.degrees(dec_target), full_mu_per_target, n_events_per_target_list, lambda_per_target_list, lambda_corrected_per_target_list)

    estimator_data = pd.DataFrame(estimator_data, columns=column_names)

    return estimator_data

#define the input path and save files with event samples
dec_center = np.radians(-30)
ra_center = np.radians(0)
patch_radius = np.radians(25)

input_path = './datasets/iso_samples/decCenter_%.0f' % np.degrees(dec_center)
input_filelist = []

for file in os.listdir(input_path):

    filename = os.path.join(input_path, file)

    if os.path.isfile(filename) and '%.0f' % np.degrees(patch_radius) in filename:

        input_filelist.append(filename)

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#define the maximum zenith angle
theta_max = np.radians(80)

#define the observation time
time_begin = Time('2010-01-01T00:00:00', scale = 'utc', format = 'fits').gps
time_end = Time('2020-01-01T00:00:00', scale = 'utc', format = 'fits').gps
obs_time = time_end - time_begin

#defines the number of events
n_events = 100_000

#defines the maximum and minimum declinations
dec_max = dec_center + patch_radius
dec_min = dec_center - patch_radius

#defines the NSIDE parameter
NSIDE = 128
npix = hp.nside2npix(NSIDE)

#defines the target radius
target_radius = np.radians(1.5)
target_area = 2*np.pi*(1 - np.cos(target_radius))

#define the tau parameter
tau = 86_164 #in seconds

#compute the coordinates of the center of each bin and transform them into ra and dec. Compute the corresponding exposure map
all_pixel_indices = np.arange(npix)
ra_target, colat_center = get_binned_sky_centers(NSIDE, all_pixel_indices)
full_dec_target = colat_to_dec(colat_center)

#filter out bins such that targets are partially or fully outside circular patch
ra_target, dec_target = get_bins_in_patch(ra_target, full_dec_target, ra_center, dec_center, patch_radius, target_radius)

#order targets by declination
sorted_indices = dec_target.argsort()
ra_target, dec_target = ra_target[sorted_indices], dec_target[sorted_indices]

#compute the normalized exposure
unique_dec_target = np.unique(full_dec_target)
unique_exposure = compute_directional_exposure(unique_dec_target, theta_max, pao_lat)
integrated_exposure = 2*np.pi*np.trapz(unique_exposure*np.cos(unique_dec_target), x = unique_dec_target)

#compute the normalized exposure map
exposure_map = compute_directional_exposure(dec_target, theta_max, pao_lat) / integrated_exposure

#define the output path
output_path = './datasets/iso_estimators/decCenter_-30'

if not os.path.exists(output_path):
    os.makedirs(output_path)

start_all = datetime.now()

#compute the number of doublets for each sample
for i, input_file in enumerate(input_filelist):

    start = datetime.now()

    #define the name of the output file
    basename = '_'.join(np.array(os.path.basename(input_file).split('_'))[[0, 2, 3, 4, 5, 6, 7, -2, -1]])
    output_file = 'Estimators_targetRadius_%.01f_' % np.degrees(target_radius) + basename

    #save dataframe with isotropic events
    event_data = pd.read_parquet(input_file, engine = 'fastparquet')

    #compute the expected number of events in each target
    estimator_data = compute_estimators_per_target(event_data, ra_target, dec_target, target_radius, exposure_map, obs_time, n_events)

    print(estimator_data.describe())

    #save doublet data into a parquet file
    output_file = os.path.join(output_path, output_file)

    estimator_data.to_parquet(output_file, index = True)

    print('Treating %i / %i event sample took' % (i, len(input_filelist)), datetime.now() - start, 's')

print('Processing %i samples took ' % len(input_filelist), datetime.now() - start_all, 's')
