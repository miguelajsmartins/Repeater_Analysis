import numpy as np
import pandas as pd
import numpy.ma as ma

from random import seed
from datetime import datetime

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

import sys
import os

sys.path.append('../src/')

import matplotlib.pyplot as plt

from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import compute_directional_exposure

#get the principal argument for a given angle
def get_principal_argument(x):

    principal_argument = np.angle(np.exp(1j*x))

    is_positive = principal_argument >= 0

    return ma.masked_array(principal_argument, mask = np.logical_not(is_positive)).filled(fill_value = 2*np.pi + principal_argument)

#generate declination uniform on sphere
def compute_dec(u, dec_min, dec_max):

    #computes the declination over the whole sky uniformely
    dec = np.arcsin(2*u - 1)

    #keeps only events inside declination band
    in_band = np.logical_and(dec > dec_min, dec < dec_max)

    dec = ma.masked_array(dec, mask = np.logical_not(in_band)).filled(fill_value = np.nan)

    #sort dec along the columns
    dec = np.sort(dec, axis = 1)

    nan_columns = np.all(np.isnan(dec), axis = 0)

    return dec[:,np.logical_not(nan_columns)]

#compute right ascension for u uniform in [0, 1]
def compute_ra(u, dec_center, ra_center, max_ra_width, patch_radius, dec):

    #compute min right ascension. this only works if ra_center = 0
    ra_min = ra_center - .5*max_ra_width

    #initialize ra array
    ra = get_principal_argument(max_ra_width*u + ra_min)

    #compute the maximum and minimum right ascensions
    delta_ra = np.arccos( (np.cos(patch_radius) - np.sin(dec_center)*np.sin(dec)) / (np.cos(dec)*np.cos(dec_center)) )

    ra_right = get_principal_argument(ra_center + delta_ra)
    ra_left = get_principal_argument(ra_center - delta_ra)

    in_patch = np.logical_or(ra > ra_left, ra < ra_right)

    ra = ma.masked_array(ra, mask = np.logical_not(in_patch)).filled(fill_value = np.nan)
    dec = ma.masked_array(dec, mask = np.logical_not(in_patch)).filled(fill_value = np.nan)

    #clean columns such that all elements are nan
    nan_columns = np.all(np.isnan(dec), axis = 0)

    return ra[:,np.logical_not(nan_columns)], dec[:,np.logical_not(nan_columns)]

#compute right ascention limits for a given declination and a circular patch
def get_max_ra_width(dec, dec_min, dec_max, dec_center, patch_radius):

    ra_width = np.ones(len(dec))

    #computes max ra width for different cases
    dec_extreme = np.logical_or(dec == dec_max, dec == dec_min)
    dec_not_extreme = np.logical_not(dec_extreme)

    ra_width[dec_extreme] = 0
    ra_width[dec_not_extreme] = np.arccos(( np.cos(patch_radius) - np.sin(dec[dec_not_extreme])*np.sin(dec_center) ) / (np.cos(dec_center)*np.cos(dec[dec_not_extreme])))

    return 2*ra_width

#accept events given the auger instantenous acceptance
def compute_accepted_events(time, ra, dec, pao_loc, theta_max):

    #save the pao latitude
    pao_lat = pao_loc.lat.rad

    # compute local sidereal time
    lst = time.sidereal_time('mean').rad

    # compute zenith angle of event and only accept if smaller than theta_max
    theta = ang_diff(dec, pao_lat, ra, lst)

    #compute acceptance probability
    accept_prob = np.cos(theta)

    rand_number = np.random.random(time.shape)

    #accept events if they are in FoV and accoring to acceptance function
    accepted = np.logical_and(theta < theta_max, rand_number < accept_prob)

    accepted_time = ma.masked_array(time.gps, mask = np.logical_not(accepted)).filled(fill_value = np.nan)
    accepted_ra = ma.masked_array(ra, mask = np.logical_not(accepted)).filled(fill_value = np.nan)
    accepted_dec = ma.masked_array(dec, mask = np.logical_not(accepted)).filled(fill_value = np.nan)
    accepted_lst = ma.masked_array(lst, mask = np.logical_not(accepted)).filled(fill_value = np.nan)
    accepted_theta = ma.masked_array(theta, mask = np.logical_not(accepted)).filled(fill_value = np.nan)

    return accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst

if __name__ == '__main__':

    #fix seed
    seed = 47
    np.random.seed(seed)

    #set position of the pierre auger observatory
    lat_pao = np.radians(-35.15) # this is the average latitude
    long_pao = np.radians(-69.2) # this is the averaga longitude
    height_pao = 1425*u.meter # this is the average altitude

    #define the earth location corresponding to pierre auger observatory
    pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

    #define start and end times
    start_date_fits = '2010-01-01'
    end_date_fits = '2020-01-01'
    start_date = Time(start_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps
    end_date = Time(end_date_fits + 'T00:00:00', format = 'fits', scale='utc', location=pao_loc).gps

    #define the width of the patch in sky
    patch_radius = np.radians(25)

    #define the position of a source
    dec_center = np.radians(-30)
    ra_center = np.radians(0)

    #define the output path
    output_path = './datasets/iso_samples/decCenter_%.0f' % np.degrees(dec_center)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #define the number of events in the whole sky
    n_events = 100_000

    #define the maximum accepted zenith angle
    theta_min = np.radians(0)
    theta_max = np.radians(80)
    dec_max = lat_pao + theta_max

    #define the maximum and minimum declinations in the patch and the corresponding max and min uniformly distributed random numbers
    dec_bottom = dec_center - patch_radius
    dec_top = dec_center + patch_radius

    if dec_top > dec_max:
        print('please make sure all events in patch are in FoV of observatory!')
        exit()

    #compute the integrated exposure over the circular area to determine the number of expected events in the cap
    dec_in_patch = np.linspace(dec_bottom, dec_top, 1000)
    ra_width_in_patch = get_max_ra_width(dec_in_patch, dec_bottom, dec_top, dec_center, patch_radius)

    max_ra_width = max(ra_width_in_patch)

    dec_range = np.linspace(-np.pi / 2, dec_max, 1000)

    exposure_in_patch = compute_directional_exposure(dec_in_patch, theta_max, lat_pao)
    total_exposure = compute_directional_exposure(dec_range, theta_max, lat_pao)

    integrated_exposure_in_patch = np.trapz(exposure_in_patch*np.cos(dec_in_patch)*ra_width_in_patch, x = dec_in_patch)
    total_integrated_exposure = 2*np.pi*np.trapz(total_exposure*np.cos(dec_range), x = dec_range)

    n_events_accepted_in_patch = np.ceil((integrated_exposure_in_patch / total_integrated_exposure)*n_events).astype('int')

    #multiply the computed number of events by factor that takes into account the rejection of events
    print('Number of events to accept in patch %i' % n_events_accepted_in_patch)

    #defines the number of isotropically distributed events before accepting them
    n_events_in_patch = int(5*n_events_accepted_in_patch)
    #n_events_in_vertical_strip = int(20*( 2*patch_radius / np.pi )*n_events)

    #defines the number of iterations and sets up a counter
    n_iter = 10
    counter = 0

    #start of program
    start_time = datetime.now()

    for i in range(n_iter):

        #defines the number of samples
        n_samples = int(10)

        #generate vectors of uniformly distributed declination
        start_time_1 = datetime.now()

        rand_dec = np.random.random((n_samples, n_events))
        dec = compute_dec(rand_dec, dec_bottom, dec_top)

        print('Generating declinations took', start_time_1 - start_time)

        #compute right ascensions in the patch
        start_time_2 = datetime.now()

        rand_ra = np.random.random(dec.shape)
        ra, dec = compute_ra(rand_ra, dec_center, ra_center, max_ra_width, patch_radius, dec)

        print('Generating right ascensions took', start_time_2 - start_time_1)

        #generate a collection of time stamps
        start_time_3 = datetime.now()

        time = (end_date - start_date)*np.random.random(dec.shape) + start_date
        time = Time(time, format='gps', scale='utc', location=pao_loc)

        #accept events given the instaneous exposure of the observatory
        accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = compute_accepted_events(time, ra, dec, pao_loc, theta_max)

        print('Generating time-stamps and cccepting events took ', datetime.now() - start_time_3, ' s')

        #order events by time
        accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst = time_ordered_events(accepted_time, accepted_ra, accepted_dec, accepted_theta, accepted_lst)

        #save each sample as a dataframe
        for sample in range(n_samples):

            #save accepted events into a dataframe
            accepted_event_data = pd.DataFrame(zip(accepted_time[sample,:], np.degrees(accepted_ra[sample,:]), np.degrees(accepted_dec[sample,:]), np.degrees(accepted_theta[sample,:]), np.degrees(accepted_lst[sample,:])), columns=['gps_time', 'ra', 'dec', 'theta', 'lst'])

            #filter all nan values
            accepted_event_data.dropna(inplace = True, ignore_index = True)

            #choose a sample of events with only n_events_accepted_in_patch
            final_accepted_event_data = accepted_event_data.sample(n = n_events_accepted_in_patch, random_state = sample, ignore_index = True)
            final_accepted_event_data.sort_values(by='gps_time', inplace = True, ignore_index = True)

            if counter % 10 == 0:
                print(final_accepted_event_data)
                print('Produced %i / %i samples!' % (counter, n_samples*n_iter))

            #saves the dataframes in files
            output_name = 'IsoDist_%i_decCenter_%.0f_raCenter_%.0f_patchRadius_%.0f_acceptance_th80_10years_sample_%i.parquet' % (int(n_events_accepted_in_patch), np.degrees(dec_center), np.degrees(ra_center), np.degrees(patch_radius), counter)

            final_accepted_event_data.to_parquet(os.path.join(output_path, output_name), index=True)

            counter+=1

        print('Entire analsyis took ', datetime.now() - start_time, ' s')

    plt.scatter(final_accepted_event_data['ra'], final_accepted_event_data['dec'], s = 3, alpha = .5)
    plt.show()
