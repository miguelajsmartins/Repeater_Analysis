import numpy as np
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.interpolate import Akima1DInterpolator as akima_spline

#for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import sys
import os

sys.path.append('../src/')

from hist_manip import data_2_binned_errorbar
from fit_routines import perform_fit_exp

from axis_style import set_style
from axis_style import set_cb_style

#enable latex rendering and latex like style font
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

#compute n_samples of n_events distributed uniformly in time
def get_time_stamps(n_samples, n_events, obs_time):

    #compute matrix of events times
    event_time = np.random.randint(0, obs_time, size = (n_samples, n_events))

    return event_time

#compute times of events from flare
def get_flare_time_stamps(n_samples, n_events, flare_start, flare_duration):

    #compute matrix of events times
    event_time = np.random.randint(flare_start, flare_start + flare_duration, size = (n_samples, n_events))

    return event_time

#compute array given a matrix of time stamps
def compute_lambda(event_time, exp_rate):

    #compute lambda
    event_time = np.sort(event_time, axis = 1)
    time_diff = np.diff(event_time, axis = 1)*exp_rate

    lambda_array = -np.sum(np.log(time_diff), axis = 1)

    return lambda_array

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

    #print(np.exp(-tail_slope*lambda_array[lambda_above_fit_init]))
    #const = (fit_scale / tail_slope)*np.exp(-tail_slope*fit_initial)
    lambda_pvalues[lambda_above_fit_init] = (1 - discrete_cdf_lambda[-1])*np.exp(-tail_slope*(lambda_array[lambda_above_fit_init] - fit_initial)) # - fit_initial))

    #print('Discrete p_value:', np.log10(1 - discrete_cdf_lambda[-1]))
    #print('Interpolated p_value:', np.log10(1 - interpolated_cdf_lambda(fit_initial)))
    #print('Fitted p_value', np.log10(np.exp(-tail_slope*fit_initial)) )

    return lambda_pvalues

#function to print the content of each bin in 2d histogram
def print_heatmap_bin_content(ax, x_bins, y_bins, bin_content, fontsize):

    for i, x_pos in enumerate(x_bins):

        for j, y_pos in enumerate(y_bins):

            ax.text(x_pos, y_pos, r'$%.1f$' % bin_content[i, j], ha="center", va="center", fontsize = fontsize)

    return ax

if __name__ == '__main__':

    #define the input directory
    input_path = './datasets/isotropic_samples'

    #define the output directory
    output_path = './results'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #loop over the files with isotropic samples
    iso_samples_filename_list = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and 'IsoDist_LambdaPValues' in filename:

            iso_samples_filename_list.append(filename)

    #convert list into array
    iso_samples_filename_list = np.array(iso_samples_filename_list)

    #load dataframe with data from lambda distribution
    exp_nEvents = 10
    obs_time = 10 # in years

    lambda_dist_file = os.path.join(input_path, 'IsoDist_LambdaDist_mu_%i_nsamples_1000000_obsTime_%iyears.json' % (exp_nEvents, obs_time))

    if not os.path.exists(lambda_dist_file):
        print('Requested file: %s with lambda dist does not exist!' % lambda_dist_file)
        exit()

    #load the lambda_dist information
    lambda_dist_df = pd.read_json(lambda_dist_file)
    max_flare_intensity = lambda_dist_df['nevents'].min()

    #fix the seed
    seed = 20
    np.random.seed(seed)

    #define the size of each sample
    n_samples = 10_000
    pvalue_min = 1 / n_samples

    #define important quantities
    obs_time = obs_time * 86_164 * 366.24 #convert years into seconds
    exp_rate = exp_nEvents / obs_time

    #save colormap
    color_map = plt.get_cmap('RdBu_r')

    # loop over lambda dist files with different numbers of events for the same expected number of events
    for row_index, iso_nEvents_row in lambda_dist_df.iterrows():

        #if row_index > 5:
            #continue

        #save important info from each row
        iso_nEvents = iso_nEvents_row['nevents']
        poisson_pvalue = iso_nEvents_row['pvalue_poisson']
        lambda_dist_bin_centers = iso_nEvents_row['Lambda_bin_centers']
        lambda_dist_bin_content = iso_nEvents_row['Lambda_bin_content']
        lambda_dist_fit_init = iso_nEvents_row['Lambda_dist_fit_init']
        lambda_dist_fit_scale = iso_nEvents_row['Lambda_dist_fit_params'][0]
        lambda_dist_fit_tail_slope = iso_nEvents_row['Lambda_dist_fit_params'][1]

        lambda_dist_integral = np.trapz(lambda_dist_bin_content[::10], x = lambda_dist_bin_centers[::10])

        lambda_dist = [lambda_dist_bin_centers, lambda_dist_bin_content]

        print('---- Start -----')
        print('poisson_pvalue = %.4f, min_pvalue = %.4f' % (poisson_pvalue, pvalue_min))
        print('----------------')

        #does not print out maps for each poisson_pvalue < 1 / n_samples
        if poisson_pvalue < pvalue_min:
            print('pvalue smaller than 1/ n_samples. Skipping')
            continue

        #make samples with events uniformely distributed in time
        iso_time_stamps = get_time_stamps(n_samples, iso_nEvents, obs_time)

        #print(np.sort(iso_time_stamps[0, :]))

        #define a grid of events_per_flare and flare durations. selects start_time of the flare
        events_per_flare_array = np.arange(1, max_flare_intensity + 1, 1)
        flare_duration_array = np.logspace(-4, 0, 81, base = 10)*obs_time

        seed_array = np.linspace(0, iso_nEvents, iso_nEvents + 1, dtype = 'int')

        #contaminate isotropic samples with flares and compute Lambda and postrial pvalues
        #compute for each each number of events per flare and flare duration the percentage of lambda_pvalues below poisson p_value
        frac_lambdaPvalues_below_poissonPvalue = []
        all_lambda_tensor = []
        all_lambda_pvalues_tensor = []
        start_time = datetime.now()

        #this is un-inefficient!!!
        for j, events_per_flare in enumerate(events_per_flare_array):

            new_seed = seed_array[j]
            np.random.seed(new_seed)

            #remove sub-sample of times for each realization of uniformely distributed events
            new_iso_nEvents = iso_nEvents - events_per_flare

            subset_iso_time_stamps_per_sample = np.random.choice(iso_time_stamps.shape[1], size = new_iso_nEvents, replace = False)
            subset_iso_time_stamps = iso_time_stamps[:,subset_iso_time_stamps_per_sample]

            #initialize lists
            frac_lambdaPvalues_below_poissonPvalue_per_flare_intensity = []
            lambda_per_intensity = []
            lambda_pvalues_per_intensity = []

            for flare_duration in flare_duration_array:

                if obs_time - flare_duration > 0:
                    flare_start = np.random.randint(0, obs_time - flare_duration, size = 1)
                else:
                    flare_start = 0

                #generate events from flares
                flare_time_stamps = get_flare_time_stamps(n_samples, events_per_flare, flare_start, flare_duration)

                #introduce flare events into uniform time distribution
                iso_time_stamps_with_flares = np.concatenate((subset_iso_time_stamps, flare_time_stamps), axis = 1)

                #compute lambda for the new distribution
                lambda_iso_with_flares = compute_lambda(iso_time_stamps_with_flares, exp_rate)

                #compute pvalues using lambda
                lambda_pvalues_iso_with_flares = get_lambda_pvalues(lambda_iso_with_flares, lambda_dist, lambda_dist_fit_init, lambda_dist_fit_tail_slope, lambda_dist_fit_scale)

                #compute the fraction of pvalues less than the poisson pvalue
                below_poisson_pvalue = lambda_pvalues_iso_with_flares <= poisson_pvalue

                lambda_per_intensity.append(lambda_iso_with_flares)
                lambda_pvalues_per_intensity.append(lambda_pvalues_iso_with_flares)
                frac_lambdaPvalues_below_poissonPvalue_per_flare_intensity.append(len(lambda_pvalues_iso_with_flares[below_poisson_pvalue]))

            #fill list
            frac_lambdaPvalues_below_poissonPvalue.append(frac_lambdaPvalues_below_poissonPvalue_per_flare_intensity)
            all_lambda_tensor.append(lambda_per_intensity)
            all_lambda_pvalues_tensor.append(lambda_pvalues_per_intensity)

        #convert list into array
        frac_lambdaPvalues_below_poissonPvalue = np.array(frac_lambdaPvalues_below_poissonPvalue) / n_samples #- poisson_pvalue
        all_lambda_tensor = np.array(all_lambda_tensor)
        all_lambda_pvalues_tensor = np.array(all_lambda_pvalues_tensor)

        #compute pos-trial pvalue
        postrial_lambda_pvalues = np.ones(frac_lambdaPvalues_below_poissonPvalue.shape)
        non_unitary_frac = frac_lambdaPvalues_below_poissonPvalue != 1

        postrial_lambda_pvalues[non_unitary_frac] = 1 - frac_lambdaPvalues_below_poissonPvalue[non_unitary_frac]
        postrial_lambda_pvalues[np.logical_not(non_unitary_frac)] = pvalue_min

        print('The analysis took', datetime.now() - start_time, 's to run!')

        #start counting time for plotting
        start_time = datetime.now()

        #----------------------------
        # Plotting
        #----------------------------
        #initialize figure and axis
        fig_lambda_performance = plt.figure(figsize=(10, 4))

        ax_lambda_performance = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2, fig=fig_lambda_performance)
        ax_lambda_dist_1day = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1, fig=fig_lambda_performance)
        ax_lambda_dist_1month = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1, fig=fig_lambda_performance)
        ax_lambda_pvalue_dist_1day = plt.subplot2grid((2, 4), (1, 2), colspan=1, rowspan=1, fig=fig_lambda_performance)
        ax_lambda_pvalue_dist_1month = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1, fig=fig_lambda_performance)

        #compute lambda distributions for different flare intensities and times
        duration_1day = 86_164
        duration_1month = 30*86_164

        lambda_bins = 100
        lambda_pv_bins = 50
        lambda_lower = lambda_dist_bin_centers[0]
        lambda_upper = lambda_dist_bin_centers[-1]

        color_list = color_map(np.linspace(0.2, .9, len(events_per_flare_array[::2])))

        for j, flare_intensity in enumerate(events_per_flare_array[::2]):

            #find the array of lambda and lambda_pvalues for each intensity and duration
            lambda_dist_1day = all_lambda_tensor[j, np.searchsorted(flare_duration_array, duration_1day),:]
            lambda_dist_1month = all_lambda_tensor[j, np.searchsorted(flare_duration_array, duration_1month),:]

            lambda_pvalue_dist_1day = all_lambda_pvalues_tensor[j, np.searchsorted(flare_duration_array, duration_1day),:]
            lambda_pvalue_dist_1month = all_lambda_pvalues_tensor[j, np.searchsorted(flare_duration_array, duration_1month),:]
            #
            # print('---- flare intensity = %i ----' % flare_intensity)
            # print('Flare duration =', flare_duration_array[np.searchsorted(flare_duration_array, duration_1day)])
            # print(lambda_pvalue_dist_1day.shape,
            # ', min =', lambda_pvalue_dist_1day.min(),
            # ', max =', lambda_pvalue_dist_1day.max(),
            # ', below poisson_pvalue = ', len(lambda_pvalue_dist_1day[lambda_pvalue_dist_1day < poisson_pvalue]),
            # ', below 0 = ', len(lambda_pvalue_dist_1day[lambda_pvalue_dist_1day < 0]),
            # ', nan = ', lambda_pvalue_dist_1day[np.isnan(lambda_pvalue_dist_1day)])

            #build distributions
            lambda_1day_bin_centers, lambda_1day_bin_content, lambda_1day_bin_error = data_2_binned_errorbar(lambda_dist_1day, lambda_bins, lambda_lower, lambda_upper, np.ones(n_samples), False)
            lambda_1month_bin_centers, lambda_1month_bin_content, lambda_1month_bin_error = data_2_binned_errorbar(lambda_dist_1month, lambda_bins, lambda_lower, lambda_upper, np.ones(n_samples), False)

            lambda_pv_1day_bin_centers, lambda_pv_1day_bin_content, lambda_pv_1day_bin_error = data_2_binned_errorbar(np.log10(lambda_pvalue_dist_1day), lambda_pv_bins, -10, 0, np.ones(n_samples), False)
            lambda_pv_1month_bin_centers, lambda_pv_1month_bin_content, lambda_pv_1month_bin_error = data_2_binned_errorbar(np.log10(lambda_pvalue_dist_1month), lambda_pv_bins, -10, 0, np.ones(n_samples), False)

            #plot distributions
            ax_lambda_dist_1day.plot(lambda_1day_bin_centers, lambda_1day_bin_content / np.trapz(lambda_1day_bin_content, x = lambda_1day_bin_centers), color = color_list[j], linewidth = 1)
            ax_lambda_dist_1month.plot(lambda_1month_bin_centers, lambda_1month_bin_content / np.trapz(lambda_1month_bin_content, x = lambda_1month_bin_centers), color = color_list[j], linewidth = 1)

            ax_lambda_pvalue_dist_1day.plot(lambda_pv_1day_bin_centers, lambda_pv_1day_bin_content / np.trapz(lambda_pv_1day_bin_content, x = lambda_pv_1day_bin_centers), color = color_list[j], linewidth = 1)
            ax_lambda_pvalue_dist_1month.plot(lambda_pv_1month_bin_centers, lambda_pv_1month_bin_content / np.trapz(lambda_pv_1month_bin_content, x = lambda_pv_1month_bin_centers), color = color_list[j], linewidth = 1)


        #plot nominal distributions
        ax_lambda_dist_1day.plot(lambda_dist_bin_centers[::10], lambda_dist_bin_content[::10] / lambda_dist_integral, color = 'gray', linestyle = 'dashed', linewidth = 1)
        ax_lambda_dist_1month.plot(lambda_dist_bin_centers[::10], lambda_dist_bin_content[::10] / lambda_dist_integral, color = 'gray', linestyle = 'dashed', linewidth = 1)

        pvalue_cont = np.linspace(-10, 0, 100)
        pvalue_pdf_cont = np.log(10)*np.power(10, pvalue_cont)
        ax_lambda_pvalue_dist_1day.plot(pvalue_cont, pvalue_pdf_cont, color = 'tab:gray', linestyle = 'dashed', linewidth = 1)
        ax_lambda_pvalue_dist_1month.plot(pvalue_cont, pvalue_pdf_cont, color = 'tab:gray', linestyle = 'dashed', linewidth = 1)

        ax_lambda_pvalue_dist_1day.vlines(np.log10(poisson_pvalue), 1e-5, 5, color = 'black', linestyle = 'dashed', linewidth = 1)
        ax_lambda_pvalue_dist_1month.vlines(np.log10(poisson_pvalue), 1e-5, 5, color = 'black', linestyle = 'dashed', linewidth = 1)

        #define the style of the plots
        ax_lambda_dist_1day.set_yscale('log')
        ax_lambda_dist_1month.set_yscale('log')

        ax_lambda_pvalue_dist_1day.set_ylim(1e-4, 5)
        ax_lambda_pvalue_dist_1month.set_ylim(1e-4, 5)
        ax_lambda_pvalue_dist_1day.set_yscale('log')
        ax_lambda_pvalue_dist_1month.set_yscale('log')

        ax_lambda_dist_1day = set_style(ax_lambda_dist_1day, r'$\Delta t_{\mathrm{flare}} = 1$ day', r'$\Lambda$', r'Prob. density', 12)
        ax_lambda_dist_1month = set_style(ax_lambda_dist_1month, r'$\Delta t_{\mathrm{flare}} = 1$ month', r'$\Lambda$', r'Prob. density', 12)
        ax_lambda_pvalue_dist_1day = set_style(ax_lambda_pvalue_dist_1day, r'', r'$\log_{10} p_{\Lambda}$', r'Prob. density', 12)
        ax_lambda_pvalue_dist_1month = set_style(ax_lambda_pvalue_dist_1month, r'', r'$\log_{10} p_{\Lambda}$', r'Prob. density', 12)

        #to plot countour curves to assess lambda performance against the simple number of events

        events_per_flare_grid, flare_duration_grid = np.meshgrid(events_per_flare_array, np.log10(flare_duration_array / obs_time))

        #print(events_per_flare_grid[0,:])
        #print(flare_duration_grid[:,0])
        lower_contours = np.arange(-4.25, np.log10(1 - poisson_pvalue), .25)
        upper_contours = np.linspace(np.log10(1 - poisson_pvalue), 0, 5) #, 0)
        contour_levels = np.concatenate((lower_contours, upper_contours))

        heatmap_lambda_performance = ax_lambda_performance.contourf(flare_duration_grid, events_per_flare_grid, np.log10(np.transpose(postrial_lambda_pvalues)), levels = contour_levels, cmap='RdBu', norm = mcolors.TwoSlopeNorm(vmin = contour_levels[0], vcenter = np.log10(1 - poisson_pvalue), vmax =contour_levels[-1]))

        #plot the contour corresponding to 0
        poisson_contour_lambda_performance = ax_lambda_performance.contour(flare_duration_grid, events_per_flare_grid, np.log10(np.transpose(postrial_lambda_pvalues)), levels = [np.log10(1 - poisson_pvalue)], colors = 'black', linestyles = 'dashed', linewidths = 1)

        #define the style of the axis
        ax_lambda_performance = set_style(ax_lambda_performance, r'$\mu = %.0f$ events, $N_{\mathrm{events}} = %.0f$, $T_{\mathrm{obs}} = 10$ years' % (exp_nEvents, iso_nEvents), r'$\log_{10} \left( \Delta t_{\mathrm{flare}} / 10 \mathrm{\,years} \right)$', r'$n_{\mathrm{events}}$', 12)
        #ax_lambda_performance = print_heatmap_bin_content(ax_lambda_performance, flare_duration_grid[:,0], events_per_flare_grid[0,:], np.transpose(frac_lambdaPvalues_below_poissonPvalue), 10)

        #create and define style of color bar
        cb_lambda_performance = fig_lambda_performance.colorbar(heatmap_lambda_performance, ax=ax_lambda_performance)
        cb_lambda_performance = set_cb_style(cb_lambda_performance, r'$\log_{10} \left( 1 - \displaystyle \frac{n_{\Lambda}}{n_{\mathrm{samples}}} \right)$', [contour_levels[0], 0], 12)

        labeled_contour_levels = np.linspace(-4, 0, 5)
        cb_lambda_performance.set_ticks(labeled_contour_levels)
        cb_lambda_performance.set_ticklabels(['%.1f' % label for label in labeled_contour_levels], fontsize = 12)

        cb_lambda_performance.ax.hlines(np.log10(1 - poisson_pvalue), 0, 1, color = 'black', linestyle = 'solid')
        #ax_lambda_performance.clabel(poisson_contour_lambda_performance, fontsize = 12, fmt='%.0f \%%')

        #plot axis with intuitive time scales
        intuitive_duration_array = np.log10(np.array([1, 7, 30, 366]) * 86_164 / obs_time) # in seconds
        intuitive_duration_array_label = ['1 day', '1 week', '1 month', '1 year']

        ax_intuitive_duration = ax_lambda_performance.twiny()

        ax_intuitive_duration.set_xlim(ax_lambda_performance.get_xlim())
        ax_intuitive_duration.set_xticks(intuitive_duration_array)
        ax_intuitive_duration.set_xticklabels(intuitive_duration_array_label)

        #plot vertical lines with the intuitive time scales
        ax_lambda_performance.vlines(x = intuitive_duration_array, ymin = events_per_flare_array[0], ymax = events_per_flare_array[-1], color = 'tab:gray', alpha = .5, linestyle = 'dashed')

        print('Plotting took', datetime.now() - start_time, 's to run!')

        #plot legend
        legend_handles = [Line2D([0], [0], color=color_list[j], linewidth=1, label=f'$n = {flare_intensity}$') for j, flare_intensity in enumerate(events_per_flare_array[::2])]
        legend_labels = [r'$n = %i$' % flare_intensity for flare_intensity in events_per_flare_array[::2]]

        fig_lambda_performance.legend(handles = legend_handles, labels = legend_labels, fontsize = 12, loc = 'upper center', ncols = 5, columnspacing = 1., handlelength = 1., handletextpad = .5, bbox_to_anchor=(0.75, 1), frameon = False)

        fig_lambda_performance.tight_layout()
        fig_lambda_performance.savefig(os.path.join(output_path, 'LambdaPValues_below_poissonPvalue_mu_%i_nevents_%i.pdf' % (exp_nEvents, iso_nEvents)))
