import numpy as np
import numpy.ma as ma
import pandas as pd

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

from scipy.stats import poisson

import sys
import os
import pickle

sys.path.append('../src/')

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import compute_directional_exposure
from event_manip import compute_lambda_correction
from event_manip import get_principal_ra

from produce_ud_events_in_patch_eqCoord import compute_accepted_events
from compute_estimators_pvalue_binned_targets import get_lambda_pvalue

#removes events outside patch in the sky
def get_events_in_patch(dec, ra, dec_center, patch_radius):

    delta_ra = np.arccos( (np.cos(patch_radius) - np.sin(dec_center)*np.sin(dec)) / (np.cos(dec)*np.cos(dec_center)) )

    ra_right = get_principal_ra(ra_center + delta_ra)
    ra_left = get_principal_ra(ra_center - delta_ra)

    #define as nan the values of ra outside of path
    in_patch = np.logical_or(ra > ra_left, ra < ra_right)

    ra = ma.masked_array(ra, mask = np.logical_not(in_patch)).filled(fill_value = np.nan)
    dec = ma.masked_array(dec, mask = np.logical_not(in_patch)).filled(fill_value = np.nan)

    return ra, dec

#contaminate a sample of isotropic skies with events from flares and keeps only events within a patch around the flare
def get_accepted_events_around_flare(ra_flare, dec_flare, begin_date, end_date, target_radius, pao_loc, patch_radius, theta_max):

    #define the matrix of flare durations and flare intensities
    max_intensity = 15
    n_intensities = max_intensity - 1
    n_durations = 19

    flare_intensity = np.linspace(2, max_intensity, n_intensities).astype('int')
    flare_duration = np.logspace(-4, 0, n_durations, base = 10)*obs_time

    #define the maximum number of events, out of which a few will be accepted, according to the flare intensity
    n_events = 200 #this must be greater than the maximum flare intensity

    #time stamps and coordinates of events from flare
    shape = (n_durations, n_events)
    gps_times = flare_duration[:,np.newaxis]*np.random.random(shape) + begin_date

    times = Time(gps_times, scale = 'utc', format = 'gps', location = pao_loc)
    dec = np.random.normal(dec_flare, target_radius, shape)
    ra = get_principal_ra(np.random.normal(ra_flare, target_radius, shape))

    #filter out events outside the patch
    ra, dec = get_events_in_patch(dec, ra, dec_flare, patch_radius)

    #accept events
    accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = compute_accepted_events(times, ra, dec, pao_loc, theta_max)

    #order events by time. Consequently, all nan values are sent to the end of the arrays
    accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst)

    #sets the probability of selecting nan values to 0 and normalized the probability
    choice_prob = np.ones(accepted_time.shape)
    is_nan = np.isnan(accepted_time)
    choice_prob = ma.masked_array(choice_prob, mask = is_nan).filled(fill_value = 0)
    choice_prob = choice_prob / np.sum(choice_prob, axis = 1)[:,np.newaxis]

    #define a matrix of flare events for each type of flare
    flare_event_times = np.full((n_intensities, n_durations, max_intensity), np.nan)
    flare_event_dec = flare_event_times.copy()
    flare_event_ra = flare_event_times.copy()
    flare_event_theta = flare_event_times.copy()
    flare_event_lst = flare_event_times.copy()

    for i, intensity in enumerate(flare_intensity):

        for j, duration in enumerate(flare_duration):

            event_indices = np.random.choice(accepted_time.shape[1], size = intensity, replace = False, p = choice_prob[j,:])

            flare_event_times[i, j, :intensity] = accepted_time[j, event_indices]
            flare_event_dec[i, j, :intensity] = accepted_dec[j, event_indices]
            flare_event_ra[i, j, :intensity] = accepted_ra[j, event_indices]
            flare_event_theta[i, j, :intensity] = accepted_theta[j, event_indices]
            flare_event_lst[i, j, :intensity] = accepted_lst[j, event_indices]

    return flare_duration, flare_intensity, np.stack([flare_event_times, flare_event_ra, flare_event_dec, flare_event_theta, flare_event_lst])

#contaminates isotropic sample with flare events for a lattice of flare types, and filter only the events in the target around the flare
# it assumes that iso_sample is already an array and that flare sample is of the form (event_attributes, n_flare_intensities, n_flare_durations, n_events_from_flare)
def get_flare_and_iso_events_around_flare(iso_sample, flare_sample, ra_flare, dec_flare, target_radius):

    #save important quantities
    n_intensities = flare_sample.shape[1]
    n_durations = flare_sample.shape[2]
    max_events_from_flare = flare_sample.shape[3]
    max_events = iso_sample.shape[1] + max_events_from_flare

    #define the number of events to remove
    n_events_from_flare = np.linspace(2, max_events_from_flare, n_intensities).astype('int')
    n_events_to_keep = iso_sample.shape[1] - n_events_from_flare

    #tile sample of isotropy for each point the flare phase-space
    tiled_iso_sample = np.tile(iso_sample, (n_intensities, n_durations, 1, 1))
    tiled_iso_sample = np.transpose(tiled_iso_sample, axes = (2, 0, 1, 3))

    #copy tiled iso sample
    contaminated_iso_sample = np.empty(tiled_iso_sample.shape)

    for i in range(n_intensities):

        for j in range(n_durations):

            for k, bg_signal in enumerate(n_events_to_keep):

                #save events per flare
                flare_signal = n_events_from_flare[k]

                #choose events from isotropic background to keep
                iso_sample_indices = np.random.choice(tiled_iso_sample.shape[-1], size = bg_signal, replace = False)

                #compute full samples
                contaminated_iso_sample[:,i,j,:bg_signal] = tiled_iso_sample[:,i,j, iso_sample_indices]
                contaminated_iso_sample[:,i,j, bg_signal:] = flare_events[:,i,j,:flare_signal]


    #filter events that are outside the target around the flare
    ang_sep = ang_diff(contaminated_iso_sample[2,:,:,:], dec_flare, contaminated_iso_sample[1,:,:,:], ra_flare)

    in_flare_target = ang_sep < target_radius

    sample_times = ma.masked_array(contaminated_iso_sample[0], mask = np.logical_not(in_flare_target)).filled(fill_value = np.nan)
    sample_ra = ma.masked_array(contaminated_iso_sample[1], mask = np.logical_not(in_flare_target)).filled(fill_value = np.nan)
    sample_dec = ma.masked_array(contaminated_iso_sample[2], mask = np.logical_not(in_flare_target)).filled(fill_value = np.nan)
    sample_theta = ma.masked_array(contaminated_iso_sample[3], mask = np.logical_not(in_flare_target)).filled(fill_value = np.nan)
    sample_lst = ma.masked_array(contaminated_iso_sample[4], mask = np.logical_not(in_flare_target)).filled(fill_value = np.nan)

    #order events in time. this pushes all the nan values to the end
    sorted_indices = sample_times.argsort(axis = 2)

    sorted_times = np.take_along_axis(sample_times, sorted_indices, axis = 2)
    sorted_ra = np.take_along_axis(sample_ra, sorted_indices, axis = 2)
    sorted_dec = np.take_along_axis(sample_dec, sorted_indices, axis = 2)
    sorted_theta = np.take_along_axis(sample_theta, sorted_indices, axis = 2)
    sorted_lst = np.take_along_axis(sample_lst, sorted_indices, axis = 2)

    #if all values are nan for a given index, accross the flare-phase space, then eliminate them
    all_nan = np.all(np.isnan(sorted_times), axis = (0, 1))

    sorted_times = sorted_times[:,:,np.logical_not(all_nan)]
    sorted_dec = sorted_dec[:,:,np.logical_not(all_nan)]
    sorted_ra = sorted_ra[:,:,np.logical_not(all_nan)]
    sorted_theta = sorted_theta[:,:,np.logical_not(all_nan)]
    sorted_lst = sorted_lst[:,:,np.logical_not(all_nan)]

    return tiled_iso_sample, np.stack([sorted_times, sorted_dec, sorted_ra, sorted_theta, sorted_lst])

#compute estimators for each contaminated sample with a grid of flare configurations
def compute_estimators_in_flare_target(event_data, mu_at_flare, obs_time):

    #save event times. Note that events are already ordered in time
    event_times = event_data[0]

    #save expected rate at flare
    rate_at_flare = mu_at_flare / obs_time

    #count the number of events for each flare configuration
    is_nan = np.isnan(event_times)
    n_events_at_flare = np.count_nonzero(ma.masked_array(event_times, mask = is_nan).filled(fill_value = 0), axis = 2)

    #compute estimators for each flare type
    time_diff = np.diff(event_times, axis = 2)
    time_diff = time_diff*rate_at_flare

    #compute lambda and corrected lambda
    lambda_value = -np.nansum(np.log(time_diff), axis = 2)
    corrected_lambda_value = lambda_value - compute_lambda_correction(n_events_at_flare, mu_at_flare)

    #save number of events and lambda estimations
    estimator_data = np.stack([n_events_at_flare, lambda_value, corrected_lambda_value])

    return estimator_data

#compute pvalues for each of the estimators
def compute_estimators_pvalues(estimator_data, lambda_dist, corrected_lambda_dist, mu_at_flare):

    #given the expected number of events in each target
    mu_bins = np.append(lambda_dist['mu_low_edges'].to_numpy(), np.array(lambda_dist['mu_upper_edges'].values)[-1])
    index = np.searchsorted(mu_bins, mu_at_flare) - 1

    pvalues_lambda = get_lambda_pvalue(index, lambda_dist, estimator_data[1])
    pvalues_corrected_lambda = get_lambda_pvalue(index, corrected_lambda_dist, estimator_data[2])
    pvalues_poisson = 1 - .5*(poisson.cdf(estimator_data[0] - 1, mu_at_flare) + poisson.cdf(estimator_data[0], mu_at_flare))

    #save the pvalue data
    pvalue_data = np.stack([pvalues_poisson, pvalues_lambda, pvalues_corrected_lambda])

    return pvalue_data

#define the main fuction
if __name__ == '__main__':

    #fix seed
    seed = 47
    np.random.seed(seed)

    #define important quantities
    dec_center = np.radians(-30)
    ra_center = np.radians(0)
    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)
    n_events = 100_000

    #define the input path and save files with event samples
    input_path = './datasets/iso_samples/decCenter_%.0f' % np.degrees(dec_center)
    input_filelist = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and '%.0f' % np.degrees(patch_radius) in filename:

            input_filelist.append(filename)

    #read files with the lambda distribution per expected events
    lambda_dist_path = './datasets/lambda_dist'

    file_lambda_dist = 'CDF_GaussianKernelEstimated_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_100.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    file_corrected_lambda_dist = 'CDF_GaussianKernelEstimated_Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f_samples_100.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    file_lambda_dist = os.path.join(lambda_dist_path, file_lambda_dist)
    file_corrected_lambda_dist = os.path.join(lambda_dist_path, file_corrected_lambda_dist)

    #check if both requested file exist
    if (not os.path.exists(file_lambda_dist)) or (not os.path.exists(file_lambda_dist)):
        print('One of the requested files does not exist!')
        exit()

    #save dataframes with the lambda distribution
    lambda_dist = pd.read_json(file_lambda_dist)
    corrected_lambda_dist = pd.read_json(file_corrected_lambda_dist)

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

    #compute the expected number of events in the position of the flare
    dec_array = np.linspace(-np.pi / 2, np.pi / 2 , 1000)
    exposure_array = compute_directional_exposure(dec_array, theta_max, pao_lat)
    integrated_exposure = 2*np.pi*np.trapz(exposure_array*np.cos(dec_array), x = dec_array)

    target_area = 2*np.pi*(1 - np.cos(target_radius))
    exposure_at_flare = compute_directional_exposure([dec_center], theta_max, pao_lat) / integrated_exposure
    mu_at_flare = n_events*exposure_at_flare*target_area
    mu_at_flare = mu_at_flare[0]

    print('Expected number of events in flare target: mu = %.2f' % mu_at_flare)

    #make samples with a grid of flares with different durations and intensities
    start = datetime.now()

    #initialize lists to save all contaminated and iso samples
    estimator_data_flare_samples = []
    estimator_data_iso_samples = []

    pvalues_data_flare_samples = []
    pvalues_data_iso_samples = []

    for i, file in enumerate(input_filelist[:10]):

        #save corresponding dataframe and convert to array
        iso_sample = pd.read_parquet(file, engine = 'fastparquet')

        iso_sample['dec'] = np.radians(iso_sample['dec'])
        iso_sample['ra'] = np.radians(iso_sample['ra'])
        iso_sample['theta'] = np.radians(iso_sample['theta'])
        iso_sample['lst'] = np.radians(iso_sample['lst'])

        iso_sample = iso_sample.to_numpy().T

        #produce flare events
        flare_duration, flare_intensity, flare_events = get_accepted_events_around_flare(ra_center, dec_center, time_begin, time_end, target_radius, pao_loc, patch_radius, theta_max)

        #contaminate isotropic samples with events from flare and filter events coming from target around flare
        iso_events, events_around_flare = get_flare_and_iso_events_around_flare(iso_sample, flare_events, ra_center, dec_center, target_radius)

        #compute estimators for each sample per flare type, for contaminated and isotropic samples
        iso_estimators_data = compute_estimators_in_flare_target(iso_events, mu_at_flare, obs_time)
        flare_estimators_data = compute_estimators_in_flare_target(events_around_flare, mu_at_flare, obs_time)

        #compute the pvalues for each sample of flares
        iso_pvalues_data = compute_estimators_pvalues(iso_estimators_data, lambda_dist, corrected_lambda_dist, mu_at_flare)
        flare_pvalues_data = compute_estimators_pvalues(flare_estimators_data, lambda_dist, corrected_lambda_dist, mu_at_flare)

        estimator_data_iso_samples.append(iso_estimators_data)
        estimator_data_flare_samples.append(flare_estimators_data)
        pvalues_data_iso_samples.append(iso_pvalues_data)
        pvalues_data_flare_samples.append(flare_pvalues_data)

        print('%i / %i files processed!' % (i, len(input_filelist)))

    print('Processing all samples took', datetime.now() - start)

    #save the combined data for the samples analysed
    flare_intensity, flare_duration = np.meshgrid(flare_intensity, flare_duration)

    estimator_data_iso_samples = np.array(estimator_data_iso_samples)
    estimator_data_flare_samples = np.array(estimator_data_flare_samples)

    pvalues_data_iso_samples = np.array(pvalues_data_iso_samples)
    estimator_data_flare_samples = np.array(estimator_data_flare_samples)

    #define the name of the output files
    output_flare_info = 'Info_FlareLattice_patchRadius_%.0f_targetRadius_%.1f_samples_%i.pkl' % (np.degrees(patch_radius), np.degrees(target_radius), len(input_filelist))
    output_flare_estimators = 'Estimators_FlareLattice_patchRadius_%.0f_targetRadius_%.1f_samples_%i.pkl' % (np.degrees(patch_radius), np.degrees(target_radius), len(input_filelist))
    output_flare_pvalues = 'PValues_FlareLattice_patchRadius_%.0f_targetRadius_%.1f_samples_%i.pkl' % (np.degrees(patch_radius), np.degrees(target_radius), len(input_filelist))
    output_iso_estimators = 'Estimators_IsoDist_patchRadius_%.0f_targetRadius_%.1f_samples_%i.pkl' % (np.degrees(patch_radius), np.degrees(target_radius), len(input_filelist))
    output_iso_pvalues = 'PValues_IsoDist_patchRadius_%.0f_targetRadius_%.1f_samples_%i.pkl' % (np.degrees(patch_radius), np.degrees(target_radius), len(input_filelist))

    #define the output path
    output_path = './datasets/flare_lattice_study'

    #save files with the results
    with open(os.path.join(output_path, output_flare_info), 'wb') as file:
        pickle.dump((flare_intensity, flare_duration), file)

    with open(os.path.join(output_path, output_flare_estimators), 'wb') as file:
        pickle.dump(estimator_data_flare_samples, file)

    with open(os.path.join(output_path, output_flare_pvalues), 'wb') as file:
        pickle.dump(pvalues_data_flare_samples, file)

    with open(os.path.join(output_path, output_iso_estimators), 'wb') as file:
        pickle.dump(estimator_data_iso_samples, file)

    with open(os.path.join(output_path, output_iso_pvalues), 'wb') as file:
        pickle.dump(pvalues_data_iso_samples, file)
