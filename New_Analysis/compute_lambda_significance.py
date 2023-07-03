import numpy as np
import pandas as pd
import healpy as hp
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
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map

#computes p value given lambda and declination band
def compute_p_value(lambda_dist_data, lambda_value, dec):

    lambda_dist_data = lambda_dist_data[ (lambda_dist_data['dec_low_edges'] <= dec) & (lambda_dist_data['dec_upper_edges'] > dec) ]

    lambda_dist_data = lambda_dist_data.reset_index()
    lambda_bin_centers = np.array(lambda_dist_data['lambda_bin_centers'].loc[0])
    lambda_bin_content = np.array(lambda_dist_data['lambda_bin_content'].loc[0])

    above_lambda = lambda_bin_centers > lambda_value

    p_value = sum(lambda_bin_content[above_lambda]) / sum(lambda_bin_content)

    return p_value

#converts colatitude to declination
def compute_lambda_significance(lambda_dist_data, filelist):

    #compute_p_value(lambda_dist_data, -1, -30)
    start_time = datetime.now()

    for i, file in enumerate(filelist[0:30]):

         data = pd.read_parquet(file, engine='fastparquet')

         if 'lambda_p_value' in data.columns:
             continue

         data['lambda_p_value'] = data.apply(lambda x: compute_p_value(lambda_dist_data, x['lambda'], x['dec_center']) if not math.isnan(x['lambda']) else np.nan, axis = 1)

         print(i, 'files done in', datetime.now() - start_time, 's')

         data.to_parquet(file, index = True)

         #print('Done')


#save names of files containing events
path_to_files = './datasets/estimators/'
filelist = []

# Loop over files in the directory
for filename in os.listdir(path_to_files):

    f = os.path.join(path_to_files, filename)

    if os.path.isfile(f) and 'Estimator' in f: # and 'Scrambled' not in f:

        filelist.append(f)

#load file with distribution of lambda for each declination
lambda_dist_data = pd.read_json('./datasets/estimator_dist/Lambda_dist_per_dec_371.json')

#computes the significance of the lambda for each pixel
compute_lambda_significance(lambda_dist_data, filelist)


# #load events from isotropic distribution
# if (len(sys.argv) == 1):
#     print('Must give a file containing distribution of events')
#     exit()
#
# #save file name
# filename = sys.argv[1]
#
# #checks if file exists
# if not os.path.exists(filename):
#     print("Requested file does not exist. Aborting")
#     exit()
#
# #save path to file and its basename
# path_input = os.path.dirname(filename)
# basename = os.path.splitext(os.path.basename(filename))[0]
# path_output = './' + path_input.split('/')[1] + '/estimators'
#
# #save dataframe with isotropic events
# event_data = pd.read_parquet(filename, engine = 'fastparquet')
#
# #compute number of events and observation time
# n_events = len(event_data.index)
# obs_time = event_data['gps_time'].loc[len(event_data.index) - 1] - event_data['gps_time'].loc[0]
#
# #set position of the pierre auger observatory
# pao_lat = np.radians(-35.15) # this is the average latitude
# pao_long = np.radians(-69.2) # this is the averaga longitude
# pao_height = 1425*u.meter # this is the average altitude
#
# #define the earth location corresponding to pierre auger observatory
# pao_loc = EarthLocation(lon=pao_long*u.rad, lat=pao_lat*u.rad, height=pao_height)
#
# #defines the maximum declination and a little tolerance
# theta_max = np.radians(80)
# dec_max = pao_lat + theta_max
#
# #defines the NSIDE parameter
# NSIDE = 64
#
# #defines the time window
# time_window = [86_164, 7*86_164, 30*86_164] #consider a time window of a single day
#
# #compute event average event rate in angular window
# ang_window = hp.nside2resol(NSIDE)
# rate = (n_events / obs_time)
#
# #computing estimators for a given skymap of events
# start_time = datetime.now()
#
# estimator_data = compute_estimators(event_data, NSIDE, rate, time_window, theta_max, pao_lat)
#
# end_time = datetime.now() - start_time
#
# print('Computing estimators took', end_time,'s')
#
# #order events by declination
# estimator_data = estimator_data.sort_values('dec_center')
# estimator_data = estimator_data.reset_index(drop=True)
#
# #prints table with summary of data
# print(estimator_data.describe())
#
# #saves estimator data
# estimator_data.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
