import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.interpolate import Akima1DInterpolator as akima_spline

import sys
import os

sys.path.append('./src/')

from hist_manip import data_2_binned_errorbar
from fit_routines import perform_fit_exp

import matplotlib.pyplot as plt

#compute array of lambda values given a n_samples with n_events each
def compute_lambda(n_samples, n_events, obs_time, exp_rate):

    #compute matrix of events times
    event_time = np.random.randint(0, obs_time, size = (n_samples, n_events))

    #compute lambda
    event_time = np.sort(event_time, axis = 1)
    time_diff = np.diff(event_time, axis = 1)*exp_rate

    lambda_array = -np.sum(np.log(time_diff), axis = 1)

    return lambda_array

#compute the distribution of lambda
def get_fitted_lambda_distribution(lambda_array):

    #compute the distribution of lambda
    lambda_lower = -5
    lambda_upper = 60
    n_bins = 100*(lambda_upper - lambda_lower) + 1 #the binning must be fine enough to avoid skewed distributions of p-values
    lambda_bin_edges = np.linspace(lambda_lower, lambda_upper, n_bins)

    #get the binned lambda_distribution
    lambda_bin_centers, lambda_bin_content, lambda_bin_error = data_2_binned_errorbar(lambda_array, n_bins, lambda_lower, lambda_upper, np.ones(len(lambda_array)), False)

    new_lambda_bin_content = np.zeros(n_bins)
    new_lambda_bin_content[0:] = lambda_bin_content

    #get the 95 % quantile
    fit_initial = np.quantile(lambda_array, .95)

    #fit the lambda distribution
    fit_result = perform_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, fit_initial)

    #save the slope of tail
    tail_slope = fit_result[0][1]
    fit_scale = fit_result[0][0]
    new_fit_init = fit_result[2][0]

    return [lambda_bin_centers, lambda_bin_content, lambda_bin_error], new_fit_init, tail_slope, fit_scale

#computes p_values for the lambda_array
def get_lambda_pvalues(lambda_array, lambda_dist, fit_initial, tail_slope, fit_scale):

    #initialize p_value array
    lambda_pvalues = np.zeros(len(lambda_array))

    #saves the lambda_bin_centers and bin_contents
    lambda_bin_centers = np.array(lambda_dist[0])
    lambda_bin_content = np.array(lambda_dist[1])

    #compute the discrete cdf of lambda and interpolate it
    below_fit_init = lambda_bin_centers <= fit_initial
    lambda_bins = lambda_bin_centers[below_fit_init]
    discrete_cdf_lambda = np.cumsum(lambda_bin_content[below_fit_init]) / np.sum(lambda_bin_content)

    interpolated_cdf_lambda = akima_spline(lambda_bins, discrete_cdf_lambda)

    #if lambda_value is below initial point the p_value used the interpolated discrete cdf
    lambda_below_fit_init = lambda_array < fit_initial
    lambda_pvalues[lambda_below_fit_init] = 1 - interpolated_cdf_lambda(lambda_array[lambda_below_fit_init])

    #if lambda_value is above initial point the p_value is analytical
    lambda_above_fit_init = lambda_array >= fit_initial
    lambda_pvalues[lambda_above_fit_init] = (fit_scale / tail_slope)*np.exp(-tail_slope*lambda_array[lambda_above_fit_init])

    return lambda_pvalues

if __name__ == '__main__':

    #define the output directory
    output_path = './datasets/isotropic_samples'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #fix the seed
    seed = 10
    np.random.seed(seed)

    #define the expected number of events
    exp_n_events = 10

    #define the duration of the experiment in seconds
    obs_time = 86_164 * 366.24 * 10
    exp_rate = exp_n_events / obs_time

    #compute the number of events needed to achieve 3-sigma detection
    n_events_max = np.ceil(exp_n_events + 3*np.sqrt(exp_n_events)).astype('int')
    n_events_min = exp_n_events #np.max([2, np.floor(exp_n_events - 3*np.sqrt(exp_n_events)).astype('int')])

    #define the number of events and corresponding poisson probability
    n_events_array = np.arange(n_events_min, n_events_max + 1, 1)

    #define the number of samples
    n_samples = 1_000_000

    #loop over the number of events
    for n_events in n_events_array:

        poisson_pvalue = 1 - .5*(poisson.cdf(n_events, exp_n_events) + poisson.cdf(n_events - 1, exp_n_events))

        #generate samples of events and compute lambda for each sample
        start_time = datetime.now()

        lambda_array = compute_lambda(n_samples, n_events, obs_time, exp_rate)

        lambda_dist, fit_init, tail_slope, fit_scale = get_fitted_lambda_distribution(lambda_array)

        print(fit_init)

        plt.plot(lambda_dist[0], lambda_dist[1], marker = 'o', markersize = 1, linestyle = 'None')
        plt.yscale('log')
        plt.savefig('test_%i.pdf' % n_events)

        lambda_pvalues = get_lambda_pvalues(lambda_array, lambda_dist, fit_init, tail_slope, fit_scale)

        #plt.hist(lambda_pvalues, bins = 100, range = [0, 1])
        #plt.show()

        print('It took', datetime.now() - start_time, 's to perform analysis for all events in sample!')

        #compute the percentage of events with p_lambda < p_poisson
        #print('Percentage of lambda p_values below poisson pvalue: %.2f' % (len(lambda_pvalues[lambda_pvalues < poisson_pvalue]) / len(lambda_pvalues)) )

    #save important info in dataframe

# #set position of the pierre auger observatory
# lat_pao = np.radians(-35.15) # this is the average latitude
# long_pao = np.radians(-69.2) # this is the averaga longitude
# height_pao = 1425*u.meter # this is the average altitude
#
# #define the earth location corresponding to pierre auger observatory
# pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)
#
# #define start and end times
# start_date_fits = '2010-01-01'
# end_date_fits = '2020-01-01'
# start_date = Time(start_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps
# end_date = Time(end_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps
#
# #define the number of events
# n_events = 2_000_000
# n_accept = int(n_events / 4)
#
# #define the maximum accepted zenith angle
# theta_max = np.radians(80)
# dec_max = lat_pao + theta_max
#
# #generate vectors of uniformly distributed
# rand_a = np.random.random(n_events)
# rand_b = np.random.random(n_events)
#
# #compute theta and phi for each value of u and v
# dec = compute_dec(rand_a, dec_max)
# ra = compute_ra(rand_b)
#
# time = np.random.randint(start_date, end_date, n_events)
# time = Time(time, format='gps', scale='utc', location=pao_loc)
#
# #accept events and save first few accepted events in file
# start = datetime.now()
#
# accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = compute_accepted_events(time, ra, dec, lat_pao, theta_max)
#
# print('Efficiency in accepting events =', len(accepted_time) / n_events)
# print('This took ', datetime.now() - start, ' s')
#
# #accept only the first 100_000 events
# accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = accepted_time[:n_accept], accepted_ra[:n_accept], accepted_dec[:n_accept], accepted_theta[:n_accept], accepted_lst[:n_accept]
#
# #order events by time
# accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst)
#
# accepted_event_data = pd.DataFrame(zip(accepted_time, np.degrees(accepted_ra), np.degrees(accepted_dec), np.degrees(accepted_theta), np.degrees(accepted_lst)), columns=['gps_time', 'ra', 'dec', 'theta', 'lst'])
#
# print(accepted_event_data)
#
# accepted_event_data.to_parquet('./datasets/UniformDist_%i_acceptance_th80_%s_%s.parquet' % (int(n_accept), start_date_fits, end_date_fits), index=True)
