import numpy as np
import pandas as pd
import healpy as hp
import math

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import scipy.interpolate as spline

import sys
import os

sys.path.append('./src/')

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import time_ordered_events
from event_manip import compute_directional_exposure
from event_manip import get_normalized_exposure_map

#computes p value given lambda and declination band
def compute_p_value(lambda_dist_data, lambda_value, rate):

    lambda_dist_data = lambda_dist_data[ (lambda_dist_data['omega_low_edges'] <= rate) & (lambda_dist_data['omega_upper_edges'] > rate) ]

    lambda_dist_data = lambda_dist_data.reset_index()

    if lambda_dist_data.empty:
        print('Dataframe is empty')
        return np.nan

    #save important variables
    lambda_bin_centers = np.array(lambda_dist_data['lambda_bin_centers'].loc[0])
    lambda_CDF_content = np.array(lambda_dist_data['cdf_lambda_bin_content'].loc[0])
    lambda_fit_tail_slope = lambda_dist_data['lambda_tail_fit_slope'].loc[0][0]

    #if lambda_value if above fit_lower_limit is uses fit to compute p_value. else it interpolates using a cubic spline to provide the p value
    if lambda_value < max(lambda_bin_centers):

        continous_CDF = spline.Akima1DInterpolator(lambda_bin_centers, lambda_CDF_content)
        p_value = 1 - continous_CDF(lambda_value)
    else:
        p_value = np.exp(-lambda_fit_tail_slope*lambda_value)

    return p_value

#converts colatitude to declination
def compute_lambda_significance(lambda_dist_data, file):

    data = pd.read_parquet(file, engine='fastparquet')

    #creates name of output file
    path = os.path.dirname(file)
    basename = os.path.basename(file)
    basename = os.path.splitext(basename)[0]

    basename_parts = basename.split("_")[:-1]
    new_basename = "_".join(basename_parts) + '_LambdaPValue.parquet'

    output_file = os.path.join(path, new_basename)

    #computes the p_value of Lambda
    start_time = datetime.now()

    data['lambda_p_value'] = data.apply(lambda x: compute_p_value(lambda_dist_data, x['lambda'], x['expected_events_in_target']) if not math.isnan(x['lambda']) else np.nan, axis = 1)

    print('Computation of lambda p values done in', datetime.now() - start_time, 's')

    data[['lambda_p_value']].to_parquet(output_file, index = True)

#if no file name is provided it complains
if len(sys.argv) < 2:
    print('Name of data set with targets must be provided. Abort!')
    exit()

#provide name of file for each to compute the lambda_significance
data_file = sys.argv[1]

#load file with distribution of lambda for each declination
lambda_dist_data = pd.read_json('./datasets/estimator_dist/Lambda_dist_per_rate_997.json')

#computes the significance of the lambda for each pixel
compute_lambda_significance(lambda_dist_data, data_file)
