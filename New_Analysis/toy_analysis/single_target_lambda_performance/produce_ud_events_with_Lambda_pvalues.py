import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.interpolate import Akima1DInterpolator as akima_spline

import sys
import os

sys.path.append('../src/')

from hist_manip import data_2_binned_errorbar
from fit_routines import perform_fit_exp

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
    tail_slope_error = fit_result[1][1]

    fit_scale = fit_result[0][0]
    fit_scale_error = fit_result[1][0]

    new_fit_init = fit_result[2][0]

    return [lambda_bin_centers, lambda_bin_content, lambda_bin_error], new_fit_init, [tail_slope, tail_slope_error], [fit_scale, fit_scale_error]

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
    lambda_pvalues[lambda_above_fit_init] = (1 - discrete_cdf_lambda[-1])*np.exp(-tail_slope*(lambda_array[lambda_above_fit_init] - fit_initial))

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
    poisson_pvalue_array = 1 - .5*(poisson.cdf(n_events_array, exp_n_events) + poisson.cdf(n_events_array - 1, exp_n_events))

    #define the number of samples
    n_samples = 1_000_000

    #initialize lists to hold lambda distributions
    lambda_bin_centers_list = []
    lambda_bin_contents_list = []
    lambda_dist_fit_init = []
    lambda_dist_fit_params = []
    lambda_dist_fit_params_error = []

    #loop over the number of events
    for n_events in n_events_array:

        #generate samples of events and compute lambda for each sample
        start_time = datetime.now()

        lambda_array = compute_lambda(n_samples, n_events, obs_time, exp_rate)

        lambda_dist, fit_init, tail_slope, fit_scale = get_fitted_lambda_distribution(lambda_array)

        #save lambda distribution
        lambda_bin_centers_list.append(lambda_dist[0])
        lambda_bin_contents_list.append(lambda_dist[1])
        lambda_dist_fit_init.append(fit_init)
        lambda_dist_fit_params.append([fit_scale[0], tail_slope[0]])
        lambda_dist_fit_params_error.append([fit_scale[1], tail_slope[1]])

        lambda_pvalues = get_lambda_pvalues(lambda_array, lambda_dist, fit_init, tail_slope[0], fit_scale[0])

        print('It took', datetime.now() - start_time, 's to perform analysis for all events in sample!')

        #save lambda values and pvalues as a dataframe
        lambda_df = pd.DataFrame(zip(lambda_array, lambda_pvalues), columns = ['lambda', 'pvalues_lambda'])

        #print df
        print(lambda_df)

        #define name of output files and saves dataframes
        output_lambda_values = 'IsoDist_LambdaPValues_mu_%i_nevents_%i_nsamples_%i_obsTime_10years.parquet' % (exp_n_events, n_events, n_samples)

        #save dataframe
        lambda_df.to_parquet(os.path.join(output_path, output_lambda_values), index = True)

    #save lambda_distributions as dataframe
    lambda_dist_df = pd.DataFrame(zip(np.full(len(n_events_array), exp_n_events), np.full(len(n_events_array), obs_time), n_events_array, poisson_pvalue_array, lambda_bin_centers_list, lambda_bin_contents_list, lambda_dist_fit_init, lambda_dist_fit_params, lambda_dist_fit_params_error),
                                  columns = ['exp_nevents', 'obs_time', 'nevents', 'pvalue_poisson', 'Lambda_bin_centers', 'Lambda_bin_content', 'Lambda_dist_fit_init', 'Lambda_dist_fit_params', 'Lambda_dist_fit_params_error'])

    #print df
    print(lambda_dist_df)

    #define name of output files and saves dataframes
    output_lambda_dist = 'IsoDist_LambdaDist_mu_%i_nsamples_%i_obsTime_10years.json' % (exp_n_events, n_samples)

    #save dataframe
    lambda_dist_df.to_json(os.path.join(output_path, output_lambda_dist), index = True)
