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
    sorted_indices = np.argsort(time, axis = 1)

    time = np.take_along_axis(time, sorted_indices, axis = 1)
    ra = np.take_along_axis(ra, sorted_indices, axis = 1)
    dec = np.take_along_axis(dec, sorted_indices, axis = 1)
    theta = np.take_along_axis(theta, sorted_indices, axis = 1)
    lst = np.take_along_axis(lst, sorted_indices, axis = 1)


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

#computes exposure given a earth location, declination, ra and maximum zenith
def compute_directional_exposure(dec, theta_max, earth_lat):

    #initialize exposure
    exposure = np.zeros(len(dec))
    hmax = np.zeros(len(dec))

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
    exposure_per_bin = compute_directional_exposure(colat_to_dec(colat), theta_max, earth_lat)

    #integrate exposure
    integrated_exposure = 2*np.pi*np.trapz(exposure_per_bin*np.sin(colat), x=colat)

    #normalize exposure
    exposure_per_bin = exposure_per_bin / integrated_exposure

    #add to map
    np.add.at(exposure_map, all_indices, exposure_per_bin)

    return exposure_map

#integrate exposure in a band between min and max declinations
def get_integrated_exposure_between(dec_min, dec_max, NSIDE, theta_max, earth_lat):

    #transform declinations to colatitude to evaluate integral
    colat_max = dec_to_colat(dec_min)
    colat_min = dec_to_colat(dec_max)

    #get exposure map
    exposure_map = get_normalized_exposure_map(NSIDE, theta_max, earth_lat)

    #vector with all indices in exposure map
    npix = hp.nside2npix(NSIDE)
    all_indices = np.arange(npix)

    #get all indices
    colat, ra = hp.pix2ang(NSIDE, all_indices)

    #keep only indices corresponding to colatitudes between requested declinations
    in_region = (colat < colat_max) & (colat > colat_min)

    return 2*np.pi*np.trapz(exposure_map[in_region]*np.sin(colat[in_region]), x=colat[in_region])

#computes the Li MA significance for a given target in a binned sky
def local_LiMa_significance(n_events, event_pixels_in_target, pixels_in_target, area_of_pixel, exposure_map, theta_max, pao_lat):

    #computes n_on and n_off
    N_on = len(event_pixels_in_target)
    N_off = n_events - N_on
    ratio_on = N_on / n_events
    ratio_off = N_off / n_events

    #computes LiMa alpha given exposures
    exposure_on = np.sum(exposure_map[pixels_in_target])*area_of_pixel
    exposure_off = np.sum(exposure_map)*area_of_pixel - exposure_on

    alpha = exposure_on / exposure_off

    #compute significance
    parcel_on = N_on * np.log((1 + 1 / alpha)*ratio_on)
    parcel_off = N_off * np.log((1 + alpha)*ratio_off)

    significance = np.sqrt(2)*np.sign(N_on - alpha*N_off)*np.sqrt(parcel_on + parcel_off)

    return exposure_on, exposure_off, significance

#computes the Li MA significance for a given target in an unbinned sky
def unbinned_local_LiMa_significance(n_events, events_in_target, area_of_target, integrated_exposure, target_dec, theta_max, pao_lat):

    #computes n_on and n_off
    N_on = len(events_in_target)
    N_off = n_events - N_on
    ratio_on = N_on / n_events
    ratio_off = N_off / n_events

    #computes LiMa alpha given exposures
    exposure_on = compute_directional_exposure([target_dec], theta_max, pao_lat)[0]*area_of_target / integrated_exposure
    exposure_off = 1 - exposure_on

    alpha = exposure_on / exposure_off

    #compute significance
    parcel_on = N_on * np.log((1 + 1 / alpha)*ratio_on)
    parcel_off = N_off * np.log((1 + alpha)*ratio_off)

    significance = np.sqrt(2)*np.sign(N_on - alpha*N_off)*np.sqrt(parcel_on + parcel_off)

    return exposure_on, exposure_off, significance
