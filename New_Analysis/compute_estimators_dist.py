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
from event_manip import time_ordered_events_ra_dec

#computes estimator dist for 1 sky sample
def compute_estimators_dist(estimator_data):

    #define bins of declination of 5 degrees of width
    dec_bin_edges = np.linspace(-90, 90, 37)

    #initialize vectors to hold values of Nmax, tau and lambda for each declination band
    tau_array = []
    n_max_array = []
    lambda_array = []

    for i in range(1, len(dec_bin_edges)):
            new_estimator_data = estimator_data[ (estimator_data['dec_center'] > dec_bin_edges[i - 1]) & (estimator_data['dec_center'] < dec_bin_edges[i])]
            tau_array.append(new_estimator_data['tau'].to_numpy())
            n_max_array.append(new_estimator_data['nMax_1day'].to_numpy())
            lambda_array.append(new_estimator_data['lambda'].to_numpy())

    #delete previous dataframe to save memory
    del estimator_data

    #create new dataframe with distributions of estimators
    data = {
        'dec_lower_edge' : dec_bin_edges[:-1],
        'dec_upper_edge' : dec_bin_edges[1:],
        'tau_dist' : tau_array,
        'nMax_1day_dist' : n_max_array,
        'lambda_dist' : lambda_array
    }

    estimator_dist = pd.DataFrame(data)

    return estimator_dist

#load events from isotropic distribution
if (len(sys.argv) < 2):
    print('Must give a file containing estimators for each pixel in the sky')
    exit()

#save file name
filename = sys.argv[1]

#checks if file exists
if not os.path.exists(filename):
    print("Requested file does not exist. Aborting")
    exit()

#checks if file containes estimators per bin
if 'Estimators' not in filename:
    print('Must provide file with estimators per pixel')
    exit()

#save path to file and its basename
path_input = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]
path_output = './' + path_input.split('/')[1] + '/estimator_dist'

#save dataframe with estimator per pixel
estimator_data = pd.read_parquet(filename, engine = 'fastparquet')

print(estimator_data)

#compute estimator dist for a given sky
estimator_dist = compute_estimators_dist(estimator_data)

#print first 10 entries of dataframe
print(estimator_dist.head(10))

#estimator_dist.to_parquet(path_output + '/' + basename + '_Estimators.parquet', index = True)
