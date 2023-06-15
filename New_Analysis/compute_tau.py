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

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#returns centers of bins in healpy sky given NSIDE
def get_binned_sky_centers(NSIDE, dec_max):

    #compute the number of pixels
    npix = hp.nside2npix(NSIDE)

    #get array with bin indexes
    bin_indexes = np.arange(npix)

    #get colatidude and ra of each bin center in helpy sky
    colat, ra = hp.pix2ang(NSIDE, bin_indexes)

    #convert to declination
    dec = colat_to_dec(colat)

    accept = dec < dec_max

    return ra[accept], dec[accept]


#computes tau for each region of healpy map
def compute_tau(event_data, ra_center, dec_center, ang_window, pao_loc):

    #save relevant quantities
    event_time = event_data['gps_time'].to_numpy()
    event_ra = np.radians(event_data['ra'].to_numpy())
    event_dec = np.radians(event_data['dec'].to_numpy())

    #delete dataframe from memory
    del event_data

    #save number of events
    n_events = len(event_time)

    #save array with 1s
    ones_n_events = np.ones(n_events)

    #compute time observation time
    obs_time = event_time[-1] - event_time[0]

    #tau array
    tau_array = []

    for i in range(len(ra_center)):

        #create arrays to compute angular difference
        dec_center_vec = dec_center[i]*ones_n_events
        ra_center_vec = ra_center[i]*ones_n_events

        #computes angular difference
        psi = ang_diff(dec_center_vec, event_dec, ra_center_vec, event_ra)

        is_in_ang_window = psi < ang_window

        #print(np.degrees(psi[is_in_ang_window]))

        #saves times of events
        times = event_time[is_in_ang_window]

        if len(times) < 2:
            tau = np.nan
        else:
            delta_times = np.diff(times)

            #compute rate of events
            rate = len(times) / obs_time

            #computes tau
            tau = rate*min(delta_times)

        #fill array with tau values
        tau_array.append(tau)

        print(i, 'events done')

    #build dataframe with tau for each pixel in healpy map
    tau_data = pd.DataFrame(zip(np.degrees(dec_center), np.degrees(ra_center), tau_array), columns = ['ra_center', 'dec_center', 'tau'])

    return tau_data

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
path_name = os.path.dirname(filename)
basename = os.path.splitext(os.path.basename(filename))[0]

#save dataframe with isotropic events
event_data = pd.read_parquet(filename, engine = 'fastparquet')

#event_data = event_data.drop(labels=range(10_000, len(event_data.index)), axis = 0)

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#defines the maximum declination
theta_max = np.radians(80)
dec_max = lat_pao + theta_max

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#defines the NSIDE parameter
NSIDE = 128

#defines the angular window
ang_window = np.radians(1)

#gets the bin_centers of the healpy sky
ra_center, dec_center = get_binned_sky_centers(NSIDE, dec_max)

print(len(ra_center))

start_time = datetime.now()

#computing tau for each event
tau_data = compute_tau(event_data, ra_center, dec_center, ang_window, pao_loc)

print('Computing tau took', datetime.now() - start_time ,'s')

print(tau_data.head())

#     #scrambling events
#     event_data = scramble_events(event_data, pao_loc)
#
#     print('Scrambling events took', datetime.now() - start_time,'s')
#
#     #save scrambled events
#     event_data.to_parquet(path_name + '/Scrambled_' + basename + '_%i.parquet' % i, index=True)
