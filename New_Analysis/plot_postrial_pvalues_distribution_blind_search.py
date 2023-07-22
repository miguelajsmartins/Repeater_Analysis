import pandas as pd
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import astropy.units as u
from astropy.coordinates import EarthLocation
import os
import sys

sys.path.append('./src/')

import hist_manip
from hist_manip import data_2_binned_errorbar
from event_manip import compute_directional_exposure
from axis_style import set_style

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#compute the distribution of pvalues for isotropic distributions
def get_postrial_pvalue_dist(filelist, pvalue_name):

    #save the distribution of p_values
    postrial_pvalues_bin_centers = []
    postrial_pvalue_dist = []

    for i, file in enumerate(filelist):

        #save data
        data = pd.read_parquet(file, engine='fastparquet')

        #save post trial pvalues and make sure null pvalues are not included
        postrial_p_values = data[pvalue_name].to_numpy()
        postrial_p_values[postrial_p_values == 0] = 1 / len(filelist)

        #get distribution of pvalues
        bin_centers, bin_content, bin_error = data_2_binned_errorbar(np.log10(postrial_p_values), 50, -3, 0, np.ones(len(data.index)), False)

        if i == 0:
            postrial_pvalue_bin_centers = bin_centers

        postrial_pvalue_dist.append(bin_content)

    #transform list into array to perform operations
    postrial_pvalue_dist = np.array(postrial_pvalue_dist)

    #compute the average and std of distribution of pvalues
    average_pvalue_dist = np.mean(postrial_pvalue_dist, axis = 0)
    std_pvalue_dist = np.std(postrial_pvalue_dist, axis = 0)

    return postrial_pvalue_bin_centers, average_pvalue_dist, std_pvalue_dist

#compute the postr trial pvalue for flare
def flare_postrial_pvalue_dist(file, n_trials, pvalue_name):

    #save data
    data = pd.read_parquet(file, engine='fastparquet')

    #print(file)
    #print(data[pvalue_name].value_counts())

    #save post trial pvalues and make sure null pvalues are not included
    postrial_p_values = data[pvalue_name].to_numpy()
    postrial_p_values[postrial_p_values == 0] = 1 / n_trials,

    #get distribution of pvalues
    bin_centers, bin_content, bin_error = data_2_binned_errorbar(np.log10(postrial_p_values), 50, -3, 0, np.ones(len(data.index)), False)

    return bin_centers, bin_content, bin_error

#compute the 2d distribution of postrial p_values
def get_2d_postrial_pvalue_dist(filelist):

    pvalue_map = []

    for i, file in enumerate(filelist):

        #save data with postrial pvalue information
        data = pd.read_parquet(file, engine='fastparquet')

        #save post trial pvalues and make sure null pvalues are not included
        postrial_poisson_pvalues = data['poisson_postrial_p_value'].to_numpy()
        postrial_lambda_pvalues = data['lambda_postrial_p_value'].to_numpy()

        postrial_poisson_pvalues[postrial_poisson_pvalues == 0] = 1 / len(filelist)
        postrial_lambda_pvalues[postrial_lambda_pvalues == 0] = 1 / len(filelist)

        #make 2d histogram
        content_poisson_lambda, lambda_edges, poisson_edges = np.histogram2d(np.log10(postrial_poisson_pvalues), np.log10(postrial_lambda_pvalues), bins=50, range=[[-3, 0], [-3, 0]])

        if i == 0:
            pvalue_map_x_edges = poisson_edges
            pvalue_map_y_edges = lambda_edges

        #save pvalues
        pvalue_map.append(content_poisson_lambda)

        if i % 100 == 0:
            print('Processed', i, '/', len(filelist), 'files')

    #transform list into array
    pvalue_map = np.array(pvalue_map)
    pvalue_map = np.sum(pvalue_map, axis = 0)

    return pvalue_map_x_edges, pvalue_map_y_edges, pvalue_map


#make directory where to save plots
basename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
split_name = basename.split('_')
output_dir = './results/' + '_'.join(split_name[1:])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#save names of files containing postrial pvalues for isotropic distributions
input_path = './datasets/estimators_binned_sky/BlindSearch_PosTrial'
filelist = []

for file in os.listdir(input_path):

    file = os.path.join(input_path, file)

    if os.path.isfile(file):
        filelist.append(file)

#save the number of trials
n_trials = len(filelist)

#save the files with the postrial pvalues for 2 samples with flare events
short_flare_path = './datasets/events_with_flares/nFlares_20_nEventsPerFlare_5_FlareDuration_1'
short_flare_file = 'BlindSearch_PosTrial_UniformDist_100000_acceptance_th80_2010-01-01_2020-01-01_nFlare_20_nEventsPerFlare_5_FlareDuration_1_PoissonPValue_nSide_64.parquet'
short_flare_file = os.path.join(short_flare_path, short_flare_file)

long_flare_path = './datasets/events_with_flares/nFlares_20_nEventsPerFlare_5_FlareDuration_7'
long_flare_file = 'BlindSearch_PosTrial_UniformDist_100000_acceptance_th80_2010-01-01_2020-01-01_nFlare_20_nEventsPerFlare_5_FlareDuration_7_PoissonPValue_nSide_64.parquet'
long_flare_file = os.path.join(long_flare_path, long_flare_file)

#compute the average and width of distribution of postrial pvalues for isotropic distributions
poisson_pt_pvalue_bin_centers, poisson_pt_pvalue_average_bin_content, poisson_pt_pvalue_std_bin_content = get_postrial_pvalue_dist(filelist, 'poisson_postrial_p_value')
lambda_pt_pvalue_bin_centers, lambda_pt_pvalue_average_bin_content, lambda_pt_pvalue_std_bin_content = get_postrial_pvalue_dist(filelist, 'lambda_postrial_p_value')

#get 2 histogram with pvalues
pvalue_map_poisson_edges, pvalue_map_lambda_edges, pvalue_map_content = get_2d_postrial_pvalue_dist(filelist)

#compute the postrial pvalues for the samples with flares
short_flare_poisson_bin_centers, short_flare_poisson_bin_content, short_flare_poisson_bin_error = flare_postrial_pvalue_dist(short_flare_file, n_trials, 'poisson_postrial_p_value')
short_flare_lambda_bin_centers, short_flare_lambda_bin_content, short_flare_lambda_bin_error = flare_postrial_pvalue_dist(short_flare_file, n_trials, 'lambda_postrial_p_value')

long_flare_poisson_bin_centers, long_flare_poisson_bin_content, long_flare_poisson_bin_error = flare_postrial_pvalue_dist(long_flare_file, n_trials, 'poisson_postrial_p_value')
long_flare_lambda_bin_centers, long_flare_lambda_bin_content, long_flare_lambda_bin_error = flare_postrial_pvalue_dist(long_flare_file, n_trials, 'lambda_postrial_p_value')

#plot the distributions of postrial pvalues
fig_postrial_pvalues_dist = plt.figure(figsize=(10, 3))

ax_poisson_pvalue = fig_postrial_pvalues_dist.add_subplot(121)
ax_lambda_pvalue = fig_postrial_pvalues_dist.add_subplot(122)
#ax_joint_pvalue_dist = fig_postrial_pvalues_dist.add_subplot(122)

#poisson postrial pvalue
ax_poisson_pvalue.plot(poisson_pt_pvalue_bin_centers, np.ones(len(poisson_pt_pvalue_bin_centers)), color = 'tab:blue')
ax_poisson_pvalue.fill_between(poisson_pt_pvalue_bin_centers, 1 - 2*poisson_pt_pvalue_std_bin_content / poisson_pt_pvalue_average_bin_content, 1 + 2*poisson_pt_pvalue_std_bin_content / poisson_pt_pvalue_average_bin_content, color = 'tab:blue', alpha = .3, label = r'$10^3$ iso. skies')
ax_poisson_pvalue.plot(short_flare_poisson_bin_centers, short_flare_poisson_bin_content / poisson_pt_pvalue_average_bin_content, color = 'tab:red', marker='o', markersize=2, label = r'$\Delta t_{\mathrm{flare}} = 1$ day')
ax_poisson_pvalue.plot(long_flare_poisson_bin_centers, long_flare_poisson_bin_content / poisson_pt_pvalue_average_bin_content, color = 'tab:orange', linestyle='none', marker='o', markersize=6, fillstyle='none', label = r'$\Delta t_{\mathrm{flare}} = 7$ days')

ax_lambda_pvalue.plot(lambda_pt_pvalue_bin_centers, np.ones(len(lambda_pt_pvalue_bin_centers)), color = 'tab:blue')
ax_lambda_pvalue.fill_between(lambda_pt_pvalue_bin_centers, 1 - 2*lambda_pt_pvalue_std_bin_content / lambda_pt_pvalue_average_bin_content, 1 + 2*lambda_pt_pvalue_std_bin_content / lambda_pt_pvalue_average_bin_content, color = 'tab:blue', alpha = .3)
ax_lambda_pvalue.plot(short_flare_lambda_bin_centers, short_flare_lambda_bin_content / lambda_pt_pvalue_average_bin_content, color = 'tab:red', marker='o', markersize=2)
ax_lambda_pvalue.plot(long_flare_lambda_bin_centers, long_flare_lambda_bin_content / lambda_pt_pvalue_average_bin_content, color = 'tab:orange', linestyle='none', marker='o', markersize=6, fillstyle='none')

#define style of axis
ax_poisson_pvalue = set_style(ax_poisson_pvalue, '', r'$\log_{10} (p_{\mathrm{poisson}}\mathrm{-value})$', r'Ratio to isotropy', 12)
ax_poisson_pvalue.set_ylim(.5, 2.5)
ax_poisson_pvalue.legend(loc='upper right', title=r'$n_{\mathrm{flares}} = 20$, $n_{\mathrm{events}} = 5$', title_fontsize=12, fontsize=12)
#ax_poisson_pvalue.set_yscale('log')

ax_lambda_pvalue = set_style(ax_lambda_pvalue, '', r'$\log_{10} (p_{\Lambda}\mathrm{-value})$', r'Ratio to isotropy', 12)
ax_lambda_pvalue.set_ylim(.5, 2.5)
#ax_lambda_pvalue.set_yscale('log')

#plot the joint dist of pvalues
#heatmap_poisson_lambda = ax_joint_pvalue_dist.pcolormesh(pvalue_map_poisson_edges, pvalue_map_lambda_edges, pvalue_map_content, norm=mcolors.LogNorm(vmin=1, vmax=pvalue_map_content.max()), cmap='Blues')

#ax_joint_pvalue_dist = set_style(ax_joint_pvalue_dist, '', r'$\log_{10} (p_{\mathrm{poisson}}\mathrm{-value})$', r'$\log_{10} (p_{\Lambda}\mathrm{-value})$', 12)

#save figure
fig_postrial_pvalues_dist.tight_layout()
fig_postrial_pvalues_dist.savefig(os.path.join(output_dir, 'distribution_postrial_pvalues_IsotropicSkies_Flares.png'), dpi=1000)
