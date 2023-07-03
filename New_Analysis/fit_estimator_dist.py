import pandas as pd
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

sys.path.append('./src/')

import hist_manip
import fit_routines
from hist_manip import data_2_binned_errorbar
from event_manip import compute_directional_exposure
from axis_style import set_style

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#rounds the declination to multiples of 5
def round_declination(dec):

    return dec - (dec % 5)

#fit the tail of the distribution of lambda and compute the corresponding expected rate
def get_fit_params_lambda_dist(filename, theta_max, lat_pao):

    #get the lambda distribution
    lambda_data = pd.read_json(complete_fname_lambda)

    #compute max declination in degrees
    dec_max = np.degrees(theta_max + lat_pao)

    #remove declinations outside field of view of observatory
    lambda_data = lambda_data[lambda_data['dec_low_edges'] < dec_max]

    #make sure lists are arrays
    lambda_data['lambda_bin_centers'] = lambda_data['lambda_bin_centers'].apply(lambda x: np.array(x))
    lambda_data['lambda_bin_content'] = lambda_data['lambda_bin_content'].apply(lambda x: np.array(x))

    #compute declination bin centers
    dec_bin_centers = lambda_data.apply(lambda x: (x['dec_low_edges'] + x['dec_upper_edges']) / 2, axis = 1).to_numpy()

    #compute the theoretical directional exposure rate for each declination bin using the average declination. This could be improved.
    theo_directional_exposure = compute_directional_exposure(np.radians(dec_bin_centers), theta_max, lat_pao)
    integrated_exposure = np.trapz(theo_directional_exposure*np.cos(np.radians(dec_bin_centers)), x=np.radians(dec_bin_centers))
    theo_directional_exposure = (theo_directional_exposure) / integrated_exposure

    #compute the error of each bin content
    lambda_data['lambda_bin_error'] = lambda_data['lambda_bin_content'].apply(lambda x: np.sqrt(x))

    #compute mean and sigma of lambda_distribution for each declination
    lambda_data['lambda_dist_mean'] = lambda_data.apply(lambda x: sum(x['lambda_bin_centers']*x['lambda_bin_content']) / sum(x['lambda_bin_content']) , axis = 1)
    lambda_data['lambda_dist_2nd_moment'] = lambda_data.apply(lambda x: sum((x['lambda_bin_centers']**2)*x['lambda_bin_content']) / sum(x['lambda_bin_content']) , axis = 1)
    lambda_data['lambda_dist_sigma'] = lambda_data.apply(lambda x: np.sqrt(x['lambda_dist_2nd_moment'] - x['lambda_dist_mean']**2) , axis = 1)

    #fit the lambda distribution as a function of declination
    lambda_data['tail_fit_params'] = lambda_data.apply(lambda x: fit_routines.perform_fit_exp(x['lambda_bin_centers'], x['lambda_bin_content'], x['lambda_bin_error'], x['lambda_dist_mean'], x['lambda_dist_sigma']), axis=1)

    #get the slope of the tail, correspoding error and chi2 of fit
    lambda_dist_tail_slope = lambda_data['tail_fit_params'].apply(lambda x: x[0][1]).to_numpy()
    lambda_dist_tail_slope_error = lambda_data['tail_fit_params'].apply(lambda x: x[1][1]).to_numpy()
    lambda_dist_tail_chi2 = lambda_data['tail_fit_params'].apply(lambda x: x[4]).to_numpy()

    #return lambda_data
    return dec_bin_centers, theo_directional_exposure, lambda_dist_tail_slope, lambda_dist_tail_slope_error, lambda_dist_tail_chi2

#get lambda distribution for specific declinations
def get_fitted_lambda_dist(filename, dec, theta_max, lat_pao):

    #get the lambda distribution
    lambda_data = pd.read_json(complete_fname_lambda)

    #compute max declination in degrees
    dec_max = np.degrees(theta_max + lat_pao)

    #remove declinations outside field of view of observatory
    lambda_data = lambda_data[lambda_data['dec_low_edges'] < dec_max]

    #rounds requested dec
    rounded_dec = round_declination(dec)

    #gives error if requested declination does not exist
    if rounded_dec not in lambda_data['dec_low_edges'].values:
        print('requested declination', dec, 'is not in datafile!')
        exit()

    #restrict to a given declination band
    lambda_data = lambda_data[lambda_data['dec_low_edges'] == rounded_dec]

    lambda_data = lambda_data.reset_index()

    #save lambda bin centers and contents
    dec_low = lambda_data['dec_low_edges'].loc[0]
    dec_high = lambda_data['dec_upper_edges'].loc[0]

    lambda_bin_centers = np.array(lambda_data['lambda_bin_centers'].loc[0])
    lambda_bin_content = np.array(lambda_data['lambda_bin_content'].loc[0])
    lambda_bin_error = np.sqrt(lambda_bin_content)

    #compute sample average and rms
    mean = sum(lambda_bin_centers*lambda_bin_content) / sum(lambda_bin_content)
    second_moment = sum((lambda_bin_centers**2)*lambda_bin_content) / sum(lambda_bin_content)
    sigma = np.sqrt(second_moment - mean**2)

    #fit lambda distribution
    popt, perr, lambda_cont, fit_curve, chi2 = fit_routines.perform_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, mean, sigma)

    #compute the cdf
    lambda_cdf = np.cumsum(lambda_bin_content) / sum(lambda_bin_content)

    #compute fitted cdf
    fitted_cdf = 1 - np.exp(-popt[1]*lambda_cont)

    return dec_low, dec_high, lambda_bin_centers, lambda_bin_content, lambda_bin_error, lambda_cont, fit_curve, lambda_cdf, fitted_cdf

#get the axis with the declination values corresponding to ticks of declination
def get_exposure_dec_axis(ax_dec, ax_exposure, nticks, theta_max, pao_lat):

    #define the axis with nticks equally spaced values of declination
    dec_max = np.degrees(theta_max + pao_lat)

    dec_max = dec_max + (dec_max % 5)

    dec_ticks = np.linspace(-90, dec_max, nticks)

    #compute the corresponding directional exposure
    dir_exposure_ticks = compute_directional_exposure(np.radians(dec_ticks), theta_max, pao_lat)

    #normalize exposure
    dec_array = np.radians(np.linspace(-90, 90, 1000))
    dir_exposure = compute_directional_exposure(dec_array, theta_max, pao_lat)
    integrated_exposure = np.trapz(dir_exposure*np.cos(dec_array), x=dec_array)

    dir_exposure_ticks = dir_exposure_ticks / integrated_exposure

    print(dir_exposure_ticks)

    #set ticks and tick labels
    ax_dec.set_xticks(dec_ticks)
    ax_dec.set_xticklabels(['%.0f' % dec for dec in dec_ticks])
    #ax_dec.margins(x=.05, y=.05)
    ax_exposure.set_xticks(dir_exposure_ticks)
    ax_exposure.set_xticklabels(['%.1f' % dir_exposure for dir_exposure in dir_exposure_ticks])
    ax_exposure.invert_xaxis()
    #ax_exposure.margins(x=0, y=0)

    return ax_dec, ax_exposure

#save file containing distribution of lambda as a function of declination
path_to_files = './datasets/estimator_dist'
fname_lambda = 'Lambda_dist_per_dec_1000.json'
complete_fname_lambda = os.path.join(path_to_files, fname_lambda)

#check if requested file exists
if not os.path.exists(complete_fname_lambda):
    print('Requested file does not exist!')
    exit()

#set position of the pierre auger observatory
lat_pao = np.radians(-35.15) # this is the average latitude
long_pao = np.radians(-69.2) # this is the averaga longitude
height_pao = 1425*u.meter # this is the average altitude

#define the earth location corresponding to pierre auger observatory
pao_loc = EarthLocation(lon=long_pao*u.rad, lat=lat_pao*u.rad, height=height_pao)

#define theta_max
theta_max = np.radians(80)

#-----------------------------------------
# Plot the distribution of lambda and its comulative, along with the corresponing fits
#-----------------------------------------
dec_list  = [-85, -45, 0, 30]
color_dec = {-85 : 'tab:red', -45 : 'tab:purple', 0 : 'tab:blue', 30 : 'darkblue'}

#to plot lambda pdf and cdf for different declinations
fig_lambda_dist = plt.figure(figsize=(10, 4))
ax_lambda_pdf = fig_lambda_dist.add_subplot(121)
ax_lambda_cdf = fig_lambda_dist.add_subplot(122)

for dec in dec_list:

    dec_low, dec_high, lambda_bin_centers, lambda_bin_content, lambda_bin_error, lambda_cont, lambda_tail_pdf, lambda_bin_cdf, fitted_cdf = get_fitted_lambda_dist(complete_fname_lambda, dec, theta_max, lat_pao)

    #for pdf
    ax_lambda_pdf.errorbar(lambda_bin_centers, lambda_bin_content, yerr=lambda_bin_error, color = color_dec[dec], alpha = .5, linewidth=1, marker='o', markersize=1, label = r'$\delta \in [%.0f^\circ, %.0f^\circ]$' % (dec_low, dec_high))
    ax_lambda_pdf.plot(lambda_cont, lambda_tail_pdf, color = color_dec[dec])

    #for cdf
    ax_lambda_cdf.errorbar(lambda_bin_centers, lambda_bin_cdf, yerr=np.sqrt(lambda_bin_cdf) / sum(lambda_bin_content), color = color_dec[dec], alpha = 0.5, markersize=1, marker = 'o')
    ax_lambda_cdf.plot(lambda_cont, fitted_cdf, color = color_dec[dec])

#lambda pdf for different declinations
ax_lambda_pdf = set_style(ax_lambda_pdf, '', r'$\Lambda$', r'Arb. units', 12)
ax_lambda_pdf.set_yscale('log')
ax_lambda_pdf.legend(loc='upper right', fontsize = 12)

#lambda cdf for different declinations
ax_lambda_cdf = set_style(ax_lambda_cdf, '', r'$\Lambda$', r'CDF($\Lambda$)', 12)

fig_lambda_dist.tight_layout()
fig_lambda_dist.savefig('./results/lambda_distribution_IsotropicSkies_th%.0f.pdf' % np.degrees(theta_max))

# ------------------------------------------
# Plot slope of lambda dist as a function of declination (and exposure) along with fit chi2
# ------------------------------------------
fig_lambda_slope = plt.figure(figsize=(10,5))
ax_lambda_slope_dir_exposure_func = fig_lambda_slope.add_subplot(121)
ax_lambda_chi2_dir_exposure_func = fig_lambda_slope.add_subplot(122)

#fit lambda distribution as a function of the declination
dec_bin_centers, theo_directional_exposure, lambda_dist_tail_slope, lambda_dist_tail_slope_error, lambda_dist_tail_chi2 = get_fit_params_lambda_dist(complete_fname_lambda, theta_max, lat_pao)

# draw slope of lambda distribution as a function of exposure and declination
ax_lambda_slope_dir_exposure_func.errorbar(theo_directional_exposure, lambda_dist_tail_slope, yerr=lambda_dist_tail_slope_error, linestyle='None', marker='o', markersize=3) #, label = r'$\delta \in [%.0f^\circ, %.0f^\circ]$' % (dec_low, dec_high))
ax_lambda_slope_dec_func = ax_lambda_slope_dir_exposure_func.twiny()

ax_lambda_slope_dec_func, ax_lambda_slope_dir_exposure_func = get_exposure_dec_axis(ax_lambda_slope_dec_func, ax_lambda_slope_dir_exposure_func, 8, theta_max, lat_pao)

ax_lambda_slope_dir_exposure_func = set_style(ax_lambda_slope_dir_exposure_func, '', r'$\omega (\delta)$', r'$\beta_{\Lambda}$', 12)
ax_lambda_slope_dec_func = set_style(ax_lambda_slope_dec_func, '', r'$\delta \;(^\circ)$', r'$\beta_{\Lambda}$', 12)

# draw chi2 of fit of lambda distribution as a function of exposure and declination
ax_lambda_chi2_dir_exposure_func.plot(theo_directional_exposure, lambda_dist_tail_chi2, linestyle='None', marker='o', markersize=3) #, label = r'$\delta \in [%.0f^\circ, %.0f^\circ]$' % (dec_low, dec_high))
ax_lambda_chi2_dec_func = ax_lambda_chi2_dir_exposure_func.twiny()

ax_lambda_chi2_dec_func, ax_lambda_chi2_dir_exposure_func = get_exposure_dec_axis(ax_lambda_chi2_dec_func, ax_lambda_chi2_dir_exposure_func, 8, theta_max, lat_pao)

ax_lambda_chi2_dir_exposure_func = set_style(ax_lambda_chi2_dir_exposure_func, '', r'$\omega (\delta)$', r'$\chi^2 / \mathrm{ndf}$', 12)
ax_lambda_chi2_dec_func = set_style(ax_lambda_chi2_dec_func, '', r'$\delta \;(^\circ)$', r'$\chi^2 / \mathrm{ndf}$', 12)

fig_lambda_slope.tight_layout()
fig_lambda_slope.savefig('./results/lambda_dist_slope_dec_func_IsotropicSkies_th%.0f.pdf' % np.degrees(theta_max))

#plt.plot(theo_directional_exposure, lambda_dist_tail_slope)
#plt.show()

#get the fit parameters
#dec_bin_centers, theo_directional_exposure, lambda_dist_tail_slope, lambda_dist_tail_slope_error, lambda_dist_tail_slope_chi2 = get_fit_params_lambda_dist(complete_fname_lambda, theta_max, lat_pao)

#lambda_bin_centers = np.array(lambda_data['lambda_bin_centers'].loc[0])
#lambda_bin_content = np.array(lambda_data['lambda_bin_content'].loc[0])
#lambda_bin_error = np.sqrt(lambda_bin_content)

#mean = sum(lambda_bin_centers*lambda_bin_content) / sum(lambda_bin_content)
#second_moment = sum((lambda_bin_centers**2)*lambda_bin_content) / sum(lambda_bin_content)
#sigma = np.sqrt(second_moment - mean**2)

#popt, perr, x, y, chi2 = fit_routines.perform_fit_exp(lambda_bin_centers, lambda_bin_content, lambda_bin_error, mean, sigma)

#print(chi2)

#plt.errorbar(theo_directional_exposure, lambda_dist_tail_slope, yerr=lambda_dist_tail_slope_error)
#plt.plot(x, y, color='tab:orange')
#plt.yscale('log')
#plt.show()
#bin_centers = lambda_data['']
