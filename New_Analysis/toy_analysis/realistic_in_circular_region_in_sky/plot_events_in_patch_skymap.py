import pandas as pd

import numpy as np
import numpy.ma as ma

import healpy as hp
from healpy.newvisufunc import projview, newprojplot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
import sys

import scipy.interpolate as spline
from scipy.stats import poisson

sys.path.append('../src/')

import hist_manip
from hist_manip import data_2_binned_errorbar
import event_manip

from event_manip import compute_directional_exposure
from event_manip import time_ordered_events
from event_manip import ang_diff
from event_manip import get_normalized_exposure_map
from event_manip import get_skymap
from event_manip import colat_to_dec
from event_manip import dec_to_colat
from event_manip import get_principal_ra

from axis_style import set_style

from array_manip import unsorted_search

#from compute_lambda_pvalue_binned_sky import compute_p_value

from compute_postrial_pvalues import get_filelist

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#convert alpha into phi for ploting
def ra_to_phi(ra):
    return np.where(ra > np.pi, ra - 2*np.pi, ra)

#compute right ascension limits for a given declination
def compute_ra_edges_per_dec(dec, ra_center, dec_center, patch_radius):

    #compute the minimum and max declination allowed to perform the calculation
    dec_min = dec_center - patch_radius
    dec_max = dec_center + patch_radius

    #mask the declination values outside the allowed declination band
    in_band = np.logical_and(dec > dec_min, dec < dec_max)

    dec = ma.masked_array(dec, mask = np.logical_not(in_band)).filled(fill_value = np.nan)

    #compute the maximum and minimum right ascensions
    delta_ra = np.arccos( (np.cos(patch_radius) - np.sin(dec_center)*np.sin(dec)) / (np.cos(dec)*np.cos(dec_center)) )

    ra_right = get_principal_ra(ra_center + delta_ra)
    ra_left = get_principal_ra(ra_center - delta_ra)

    return dec, ra_left, ra_right

#mask healpy pixels allways outside the patch
def mask_pixels_outside_patch(nside, skymap, ra_center, dec_center, patch_radius):

    #get the number of pixels
    npix = hp.nside2npix(nside)

    #get array of indices for all pixels
    indices = np.arange(npix)

    #get the colat and ra of the center of each pixel
    colat, ra = hp.pix2ang(nside, indices)

    dec, ra_left, ra_right = compute_ra_edges_per_dec(colat_to_dec(colat), ra_center, dec_center, patch_radius)

    #define the condition to be outside the patch
    outside_patch = np.logical_or(np.logical_and(ra > ra_right, ra < ra_left), np.isnan(dec))

    #mask pixels outside patch
    skymap[outside_patch] = hp.UNSEEN

    new_skymap = hp.ma(skymap)

    return new_skymap

#mask healpy pixels allways outside the field of view of the observatory
def mask_pixels_outside_fov(nside, skymap, colat_min):

    #get the number of pixels
    npix = hp.nside2npix(nside)

    #get array of indices for all pixels
    indices = np.arange(npix)

    #get the colat and ra of the center of each pixel
    colat, ra = hp.pix2ang(nside, indices)

    #mask pixels outside patch
    skymap[colat < colat_min] = hp.UNSEEN

    new_skymap = hp.ma(skymap)

    return new_skymap

def plot_skymap(skymap, fig, axis_indices, title, x_title, y_title, colormap, cb_min, cb_max, cb_label):

    hp.newvisufunc.projview(
        skymap,
        fig = fig,
        sub = axis_indices,
        #hold = True,
        override_plot_properties={'figure_size_ratio' : .6, 'cbar_pad' : 0.1},
        graticule=True,
        graticule_labels=True,
        title = title,
        xlabel=x_title,
        ylabel= y_title,
        cmap = colormap,
        cb_orientation="horizontal",
        projection_type="hammer",
        fontsize={'title':16, 'xlabel':14, 'ylabel':14, 'xtick_label':14, 'ytick_label':14, 'cbar_label' : 14, 'cbar_tick_label' : 14},
        longitude_grid_spacing = 45,
        latitude_grid_spacing = 30,
        xtick_label_color='black',
        min= cb_min,
        max= cb_max,
        unit = cb_label,
    );

#----------------------------------------------------
# Plots skymap with events in patch and exposure map
#----------------------------------------------------
if __name__ == '__main__':

    #define a few constants
    time_begin = Time('2010-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    time_end = Time('2020-01-01T00:00:00', format = 'fits', scale = 'utc').gps
    obs_time = time_end - time_begin
    obs_time_years = obs_time / (366*86_164)

    ra_center = np.radians(0)
    dec_center = np.radians(-30)
    patch_radius = np.radians(25)

    n_events = 100_000

    #set position of the pierre auger observatory
    pao_lat = np.radians(-35.15) # this is the average latitude

    #define the maximum zenith angle
    theta_max = np.radians(80)
    dec_max = pao_lat + theta_max
    colat_min = dec_to_colat(dec_max)

    #define the input directory
    input_path = './datasets/iso_samples/decCenter_%.0f' % np.degrees(dec_center)

    #define the output directory
    output_path = './results/skymaps'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #get lists of files with iso and (iso + flare) samples
    filelist_iso_samples = 'IsoDist_7235'
    filelist_iso_samples = get_filelist(input_path, filelist_iso_samples)

    #save one sample with the spatial distribution of events
    event_data = pd.read_parquet(filelist_iso_samples[0], engine = 'fastparquet')

    event_ra = np.radians(event_data['ra'])
    event_dec =  np.radians(event_data['dec'])

    #define the nside parameter
    nside = 64

    #get the skymap with the spatial distribution of events
    skymap_events_iso = get_skymap(nside, event_ra, event_dec)
    skymap_events_iso = mask_pixels_outside_patch(nside, skymap_events_iso, ra_center, dec_center, patch_radius)


    #get the skymap with the normalized exposure
    skymap_exposure = get_normalized_exposure_map(nside, theta_max, pao_lat)
    skymap_exposure = mask_pixels_outside_fov(nside, skymap_exposure, colat_min)

    #define the colormap
    colormap = plt.get_cmap('magma')

    #draw the skymap with the spatial distribution of events
    fig_event_skymap = plt.figure(figsize=(10, 4))

    plot_skymap(skymap_events_iso, fig_event_skymap, 121, '', r'$\alpha$', r'$\delta$', colormap, 0, 10, 'Number of events')

    #plot the line corresponding to the latitude of the observatory
    obs_ra_line = np.linspace(-np.pi, np.pi, 1000)
    hp.newvisufunc.newprojplot(theta=np.full(obs_ra_line.shape, dec_to_colat(pao_lat)), phi = obs_ra_line, linestyle = 'dashed', color = 'black')

    #plot the line envolving the patch
    dec_in_patch = np.linspace(-.5*np.pi, .5*np.pi, 10000)
    dec_in_patch, ra_border_left, ra_border_right = compute_ra_edges_per_dec(dec_in_patch, ra_center, dec_center, patch_radius)

    hp.newvisufunc.newprojplot(theta=dec_to_colat(dec_in_patch), phi = ra_to_phi(ra_border_left), linestyle = 'solid', color = 'white')
    hp.newvisufunc.newprojplot(theta=dec_to_colat(dec_in_patch), phi = ra_to_phi(ra_border_right), linestyle = 'solid', color = 'white')

    #draw the position of the center of the patch
    #hp.newvisufunc.newprojplot(theta=dec_to_colat(dec_center), phi = ra_to_phi(ra_center), linestyle = 'None', marker = 'x', fillstyle = 'none', color = 'white', markersize = 10)
    #plt.show()

    #plot the exposure map
    plot_skymap(skymap_exposure, fig_event_skymap, 122, '', r'$\alpha$', r'$\delta$', colormap, 0, 0.2, r'$\omega(\alpha, \delta)$')

    #print(type(var))

    #save figure
    output_event_skymap = 'Skymap_IsoDist_raCenter_%.0f_decCenter_%.0f_patchRadius_%.0f.pdf' % (np.degrees(ra_center), np.degrees(dec_center), np.degrees(patch_radius))

    fig_event_skymap.tight_layout()
    fig_event_skymap.savefig(os.path.join(output_path, output_event_skymap), dpi=1000)
