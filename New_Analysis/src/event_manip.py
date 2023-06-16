import numpy as np
import healpy as hp

from astropy.coordinates import EarthLocation

#converts colatitude to declination
def colat_to_dec(colat):
    return np.pi / 2 - colat

#converts declination into colat
def dec_to_colat(dec):
    return np.pi / 2 - dec

#order events by time
def time_ordered_events(time, ra, dec, theta, lst):

    #indices of ordered time array
    sorted_indices = time.argsort()

    time = time[sorted_indices]
    ra = ra[sorted_indices]
    dec = dec[sorted_indices]
    theta = theta[sorted_indices]
    lst = lst[sorted_indices]

    return time, ra, dec, theta, lst

#time ordering events only taking time, ra and dec
def time_ordered_events_ra_dec(time, ra, dec):

    #indices of ordered time array
    sorted_indices = time.argsort()

    time = time[sorted_indices]
    ra = ra[sorted_indices]
    dec = dec[sorted_indices]

    return time, ra, dec

#compute angular difference between 2 events
def ang_diff(dec_1, dec_2, ra_1, ra_2):
    return np.arccos(np.cos(dec_1)*np.cos(dec_2)*np.cos(ra_1 - ra_2) + np.sin(dec_1)*np.sin(dec_2))

#computes exposure given a earth location, declination, ra and maximum zenith
def compute_directional_exposure(ra, dec, theta_max, earth_lat):

    #if ra and dec have different sizes, it complains
    if len(ra) != len(dec):
        print('Cant compute exposure if declination and right ascensions vectors have different sizes')
        exit()

    #initialize exposure
    exposure = np.zeros(len(ra))
    hmax = np.zeros(len(ra))

    #compute function
    function = (np.cos(theta_max) - np.sin(earth_lat)*np.sin(dec)) / (np.cos(earth_lat)*np.cos(dec))

    #outsize FoV of observatory
    is_outside = function > 1
    hmax[is_outside] = 0

    #region in the sky allways in field of view of observatory
    is_always_visible = function < -1
    hmax[is_always_visible] = np.pi

    #rest of sky
    otherwise = (function < 1) & (function > -1)
    hmax[otherwise] = np.arccos(function[otherwise])

    #get directional exposure
    directional_exposure = np.cos(earth_lat)*np.cos(dec)*np.sin(hmax) + hmax*np.sin(earth_lat)*np.sin(dec)

    return directional_exposure

#computes the exposure map given a sky binning NSIDE
def get_normalized_exposure_map(NSIDE, theta_max, earth_lat):

    #get number of pixels
    npix = hp.nside2npix(NSIDE)

    #indices of sky
    all_indices = np.arange(npix)

    #initialize map
    exposure_map = np.zeros(npix)

    #get angles at center of each pixel
    colat, ra = hp.pix2ang(NSIDE, all_indices)

    #compute exposure for each pixel
    exposure_per_bin = compute_directional_exposure(ra, colat_to_dec(colat), theta_max, earth_lat)

    #normalize exposure
    exposure_per_bin = exposure_per_bin / sum(exposure_per_bin)

    #add to map
    np.add.at(exposure_map, all_indices, exposure_per_bin)

    return exposure_map

#draw skymap of distribution of events
def get_skymap(NSIDE, ra, dec):

    #compute the number of pixels
    npix = hp.nside2npix(NSIDE)

    #initialize map
    sky_map = np.zeros(npix)

    #convert dec to colat
    colat = dec_to_colat(dec)

    #get indices of pixels corresponding to each angle
    indices = hp.ang2pix(NSIDE, colat, ra)

    #add one per each pixel position
    np.add.at(sky_map, indices, 1)

    return sky_map
