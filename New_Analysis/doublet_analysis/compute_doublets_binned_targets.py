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

sys.path.append('./src/')

import time

from event_manip import ang_diff
from event_manip import compute_directional_exposure

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

def compute_doublets_per_target(event_data, ra_target, dec_target, target_radius, tau):

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

    #to save doublets
    doublet_per_target = []

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

        #computes the time difference between consecutive events
        time_diff = np.diff(event_time_in_target, axis=1)
        is_doublet = time_diff < tau

        #forms event doublets
        doublets = np.column_stack([
            event_time_in_target[:, :-1][is_doublet].ravel(),
            event_time_in_target[:, 1:][is_doublet].ravel(),
            event_ra_in_target[:, :-1][is_doublet].ravel(),
            event_ra_in_target[:, 1:][is_doublet].ravel(),
            event_dec_in_target[:, :-1][is_doublet].ravel(),
            event_dec_in_target[:, 1:][is_doublet].ravel()
        ])

        doublet_per_target.append(doublets)

        if i % 10 == 0:
            print('%i / %i target bands done!' % (i, len(unique_dec_target)))

    doublet_per_target = np.concatenate(doublet_per_target, axis=0)

    column_names = ['gps_time_1', 'gps_time_2', 'ra_1', 'ra_2', 'dec_1', 'dec_2']

    doublet_data = pd.DataFrame(doublet_per_target, columns=column_names)

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
NSIDE = 128
npix = hp.nside2npix(NSIDE)

#defines the target radius
target_radius = np.radians(1.5)
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

start_all = datetime.now()

#compute the number of doublets for each sample
for input_file in input_filelist[:1]:

    start = datetime.now()

    #save path to file and its basename
    basename = os.path.basename(input_file)

    #save dataframe with isotropic events
    event_data = pd.read_parquet(input_file, engine = 'fastparquet')

    #compute number of events and observation time
    n_events = len(event_data.index)
    obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]

    #compute the expected number of events in each target
    mu_per_target = n_events*exposure_map*target_area
    rate_per_target = mu_per_target

    doublet_data = compute_doublets_per_target(event_data, ra_target, dec_target, target_radius, tau)

    #clean nan and duplicated events
    doublet_data.dropna(inplace = True, ignore_index = True)
    doublet_data.drop_duplicates(inplace = True, ignore_index = True)

    #convert angles to degrees
    doublet_data[['ra_1', 'ra_2', 'dec_1', 'dec_2']] = np.degrees(doublet_data[['ra_1', 'ra_2', 'dec_1', 'dec_2']])

    print(doublet_data)
    
    #save doublet data into a parquet file
    output_file = os.path.join(output_path, 'Doublets_binnedTargetCenters_' + basename)
    doublet_data.to_parquet(output_file, index = True)

    print('Treating 1 event sample took', datetime.now() - start, 's')

print('Processing %i samples took ' % len(input_filelist), datetime.now() - start_all, 's')
