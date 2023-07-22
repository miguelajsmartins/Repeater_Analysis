import numpy as np
import pandas as pd
import healpy as hp
import math

from scipy.stats import poisson
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
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map
from event_manip import local_LiMa_significance

#converts colatitude to declination
def colat_to_dec(colat):
	return np.pi / 2 - colat

#converts declination into colat
def dec_to_colat(dec):
	return np.pi / 2 - dec

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE, bin_indexes):

	#get colatidude and ra of each bin center in helpy sky
	colat, ra = hp.pix2ang(NSIDE, bin_indexes)

	return ra, colat

#computes tau for each region of healpy map
def compute_estimators(event_data, NSIDE, rate, target_radius, time_window, theta_max, pao_lat):

	#save relevant quantities
	event_time = event_data['gps_time'].to_numpy()
	event_ra = np.radians(event_data['ra'].to_numpy())
	event_dec = np.radians(event_data['dec'].to_numpy())
	event_colat = dec_to_colat(event_dec)

	#delete dataframe from memory
	del event_data

	#compute the number of pixels and number of events
	npix = hp.nside2npix(NSIDE)
	n_events = len(event_dec)

	#compute the area of each target
	area_of_target = 2*np.pi*(1 - np.cos(target_radius))

	#define the maximum colat
	dec_max = theta_max + pao_lat

	#get exposure map
	exposure_map = area_of_target*get_normalized_exposure_map(NSIDE, theta_max, pao_lat)

	#array with all pixel indices and event pixel indices
	all_pixel_indices = np.arange(npix)
	#all_event_pixel_indices = hp.ang2pix(NSIDE, event_colat, event_ra)

	#save ra and colatitude from the center of pixels
	ra_target, colat_target = get_binned_sky_centers(NSIDE, all_pixel_indices)
	colat_min = dec_to_colat(dec_max)

	#exclude targets such that colat + target radius is > colat_max and save corresponding pixels
	fiducial_region = colat_target > colat_min + target_radius
	ra_target, colat_target = ra_target[fiducial_region], colat_target[fiducial_region]
	pixel_indices_in_fov = hp.ang2pix(NSIDE, colat_target, ra_target)

	#initialize arrays
	lambda_array = []
	events_in_target_array = []
	expected_events_in_target_array = []

	for i, colat_center in enumerate(colat_target):

		#save corresponding ra
		ra_center = ra_target[i]

		#find pixel index
		target_index = hp.ang2pix(NSIDE, colat_center, ra_center)

		#find corresponding declination
		dec_center = colat_to_dec(colat_center)

		#filter events outside a band with width 2*target_radius around targets declination
		in_band = np.abs(event_dec - dec_center) < target_radius

		events_in_band_dec = event_dec[in_band]
		events_in_band_ra = event_ra[in_band]
		events_in_band_time = event_time[in_band]

		#compute angular diferences between events in band and target
		ang_diffs = ang_diff(events_in_band_dec, dec_center, ra_center, events_in_band_ra)

		#filter events in target
		in_target = (ang_diffs < target_radius) & (ang_diffs > 0)

		#save times of events in target
		events_in_target_time = events_in_band_time[in_target]

		#save actual and expected number of events in target
		events_in_target = len(events_in_target_time)
		expected_events_in_target = n_events*exposure_map[target_index]


		if events_in_target <= 1:
			lambda_estimator = np.nan

		else:

			#compute time differences
			delta_times = np.diff(events_in_target_time)

			#computes tau and lambda
			local_rate = rate*exposure_map[target_index]
			lambda_estimator = -np.sum(np.log(delta_times*local_rate))

		#fill array with estimator values
		events_in_target_array.append(events_in_target)
		expected_events_in_target_array.append(expected_events_in_target)
		lambda_array.append(lambda_estimator)

		if ( i % 1000 == 0):
			print(i, '/', len(pixel_indices_in_fov), 'targets done!')

	#compute the p_values considering a poisson distribution
	events_in_target_array = np.array(events_in_target_array)
	expected_events_in_target_array = np.array(expected_events_in_target_array)
	lambda_array = np.array(lambda_array)
	poisson_p_value = 1 - .5*(poisson.cdf(events_in_target_array - 1, expected_events_in_target_array) + poisson.cdf(events_in_target_array, expected_events_in_target_array))

	print(poisson_p_value)
	dec_target = colat_to_dec(colat_target)

	#build dataframe with tau for each pixel in healpy map
	estimator_data = pd.DataFrame(zip(np.degrees(ra_target), np.degrees(dec_target), events_in_target_array, expected_events_in_target_array, lambda_array, poisson_p_value), columns = ['ra_target', 'dec_target', 'events_in_target', 'expected_events_in_target', 'lambda', 'poisson_p_value'])

	return estimator_data

#load events from isotropic distribution
if (len(sys.argv) == 1):
	print('Must give a file containing distribution of events')
	exit()

#save file name
filename = sys.argv[1]

#checks if file exists
if not os.path.exists(filename):
	print("Requested file does not exist. Aborting")
	exit()

#save path to file and its basename
path_input = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]
#path_output = './' + path_input.split('/')[1] + '/estimators'

#save dataframe with isotropic events
event_data = pd.read_parquet(filename, engine = 'fastparquet')

#compute number of events and observation time
n_events = len(event_data.index)
obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]

#set position of the pierre auger observatory
pao_lat = np.radians(-35.15) # this is the average latitude
pao_long = np.radians(-69.2) # this is the averaga longitude
pao_height = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)

#defines the maximum declination and a little tolerance
theta_max = np.radians(80)
dec_max = pao_lat + theta_max

#defines the NSIDE parameter
NSIDE = 64

#defines the time window
time_window = [86_164, 7*86_164, 30*86_164] #consider a time window of a single day, weak and month

#compute average event rate
target_radius = np.radians(1)
rate = (n_events / obs_time)

#computing estimators for a given skymap of events
start_time = datetime.now()

estimator_data = compute_estimators(event_data, NSIDE, rate, target_radius, time_window, theta_max, pao_lat)

end_time = datetime.now() - start_time

print('Computing estimators took', end_time,'s')

print(estimator_data.head(100))

#order events by declination
estimator_data = estimator_data.sort_values('dec_target')
estimator_data = estimator_data.reset_index(drop=True)

#prints table with summary of data
print(estimator_data.describe())

#saves estimator data
estimator_data.to_parquet(os.path.join(path_input, basename + '_PoissonPValue_nSide_%i.parquet' % NSIDE), index = True)
