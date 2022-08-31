import math
import numpy as np

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#for data manipulation
import pandas as pd

#to read files
import os

#from random methos
from random import random
from random import sample
from random import seed

#to count time
from datetime import datetime

#for statistics
from scipy import stats

#to use ROOT classes and methods
#import ROOT

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)
plt.rcParams["figure.autolayout"] = True

#Computes the average correlation function
def Average2PointCorrFunction(list_of_bin_edges, list_of_bin_contents):

    #it assumes the binning of all histograms is the same
    avg_bin_edges = list_of_bin_edges[0]

    avg_bin_content = []

    number_of_bins = len(list_of_bin_contents[0])

    for i in range(number_of_bins):

        bin_contents = []

        for hist_content in list_of_bin_contents:

            bin_contents.append(hist_content[i])

        avg_bin_content.append(np.mean(bin_contents))

    return avg_bin_content, avg_bin_edges

#Computes the dot product between 2 vectors on a unit sphere
def spherical_dot(alpha1, delta1, alpha2, delta2):
    return math.cos(delta1)*math.cos(delta2)*math.cos(alpha1 - alpha2) + math.sin(delta1)*math.sin(delta2)

#Computes the 2 point correlation function
def Get2PointCorrFunc(filename, col_alpha, col_delta, col_gpstime, max_ang_sep, SampleSize):

    df = pd.read_parquet(filename, engine = 'fastparquet')

    alpha = df[col_alpha].tolist()
    delta = df[col_delta].tolist()
    time = df[col_gpstime].tolist()

    list_of_ang_pos_all = []

    for i in range(len(alpha)):
        list_of_ang_pos_all.append([alpha[i], delta[i], time[i]])

    seed(1)

    list_of_ang_pos_sample = sample(list_of_ang_pos_all,SampleSize)

    list_of_dot_prod = []

    count = 0

    for ang_pos1 in list_of_ang_pos_sample:
        for ang_pos2 in list_of_ang_pos_sample:

            if(ang_pos1 == ang_pos2):
                continue

            ang_sep = math.acos(spherical_dot(ang_pos1[0], ang_pos1[1], ang_pos2[0], ang_pos2[1]))

            if( ang_sep > max_ang_sep):
                continue

            list_of_dot_prod.append(ang_sep)

        count+=1

        print(count,'events done!')

    #eliminate duplicate elements
    list_of_dot_prod_no_reps = np.unique(list_of_dot_prod)

    return np.degrees(list_of_dot_prod_no_reps)

#read the two point correlation function for the ensemble of uniformly distributed events
def Read2PointCorrFunc(path_to_files, max_sep, ang_window):

    #list to hold histograms 2point correlation functions
    list_of_hist_binedges_2pointcorrfunc = []
    list_of_hist_contents_2pointcorrfunc = []

    #initializes the number of pairs below
    number_of_pairs_below = []

    for file in os.listdir(path_to_files):

        filename = os.path.join(path_to_files,file)

        if os.path.isfile(filename) and '2PointCorrFunc_N_' in filename: #and len(number_of_pairs_below) < 50:

            begin_task = datetime.now()

            corrfunc = pd.read_parquet(filename, engine='fastparquet')
            corrfunc = corrfunc[corrfunc['2pointcorr_bin_edges'] <= max_sep]

            bin_edges = corrfunc['2pointcorr_bin_edges'].to_numpy()
            bin_contents = corrfunc['2pointcorr_bin_content'].to_numpy()

            list_of_hist_binedges_2pointcorrfunc.append(bin_edges)
            list_of_hist_contents_2pointcorrfunc.append(bin_contents)

            number_of_pairs_scalar = sum([bin_contents[i] for i in range(len(bin_contents)) if bin_edges[i] <= ang_window])
            number_of_pairs_below.append(int(number_of_pairs_scalar))

            print('This took', datetime.now() - begin_task)

    return number_of_pairs_below, list_of_hist_binedges_2pointcorrfunc, list_of_hist_contents_2pointcorrfunc

#define the log likelihood function
def logLikelihood(avg_bin_content, avg_bin_edges, obs_bin_content, ang_window):

    new_avg_bin_content = [avg_bin_content[i] for i in range(len(avg_bin_content)) if avg_bin_edges[i] <= ang_window]
    new_obs_bin_content = [obs_bin_content[i] for i in range(len(obs_bin_content)) if avg_bin_edges[i] <= ang_window]

    log_obs_factorial = []
    factorial_sum = []

    for obs in new_obs_bin_content:
        for j in range(1, int(obs)+1):
            log_obs_factorial.append(math.log(j))

        factorial_sum.append(sum(log_obs_factorial))

    return -np.sum(new_avg_bin_content) + np.sum(new_obs_bin_content*np.log(new_avg_bin_content)) - sum(factorial_sum)

#defines the incompatibility between distributions
def Incompatibility(estimator_list_ud, estimator_list_rep, percentile):

    #computes the mean and std of the distribution of likelihood for Isotropic dist.
    mean_ud = np.mean(estimator_list_ud)
    mean_rep = np.mean(estimator_list_rep)
    sigma_ud = math.sqrt(np.var(estimator_list_ud))

    print('Mean of test statistic for UD distribution', mean_ud)
    print('RMS of test statistic for UD distribution', sigma_ud)

    if (mean_ud < mean_rep):
        percentile = percentile
        quantile = np.quantile(estimator_list_rep, percentile)
        distance_from_mean = (quantile - mean_ud)/sigma_ud
        incompatibility_info = '%.1f percent of samples with explosions are \n %.2f sigma above mean of \n TS distribution for isotropy' % (100*(1-percentile), distance_from_mean)
        print(incompatibility_info)

    else:
        percentile = 1 - percentile
        quantile = np.quantile(estimator_list_rep, percentile)
        distance_from_mean = (quantile - mean_ud)/sigma_ud
        incompatibility_info = '%.1f percent of samples with explosions are \n %.2f sigma below mean of \n TS distribution for isotropy' % (100*percentile, distance_from_mean)
        print(incompatibility_info)

    return incompatibility_info

#computes the distribution of the estimator
# def Estimator(list_obs_bin_edges, list_obs_bin_contents, max_ang_sep):
#
#     #estimator distribution for the many isotropic realizations
#     estimator_list = []
#
#     for i in range(len(list_obs_bin_edges)):
#
#         estimator = 0
#
#         iter = 0
#
#         while obs_bin_edges[iter] < max_ang_sep:
#             estimator+=obs_bin_content[i]
#
#         estimator_list.append(estimator)
#
#     return estimator_list

#------------------------------------------
# main function
#------------------------------------------
#set path to dir with 2 point corr files
REP_DATE = '2015-01-01T00:00:00'
PERIOD_OF_REP = '86164'
N_EVENTS = '100000'
N_ACCEPTED_REP_EVENTS = '100'
N_INTENSITY = '100'
N_EXPLOSIONS = float(N_ACCEPTED_REP_EVENTS)/float(N_INTENSITY)

path_to_files_ud = '../DataSets/Vertical/UD_large_stats/2PointCorrFunc/'
path_to_files_rep = '../DataSets/Vertical/MockData_Repeaters/Repeater_FixedPosAndDate_large_stats/2PointCorrFunc/'

#sets maximum angular separation for 2-point corr. func. in degrees!
max_sep = 20 #degree
ang_window = 1 #degree

#list to hold histograms 2point correlation functions
number_of_pairs_below_ud, list_of_binEdges_2pointcorrfunc_ud, list_of_binContents_2pointcorrfunc_ud = Read2PointCorrFunc(path_to_files_ud, max_sep, ang_window)
number_of_pairs_below_rep, list_of_binEdges_2pointcorrfunc_rep, list_of_binContents_2pointcorrfunc_rep = Read2PointCorrFunc(path_to_files_rep, max_sep, ang_window)

#average 2 point correlation function
avg_bin_content_ud, avg_bin_edges_ud = Average2PointCorrFunction(list_of_binEdges_2pointcorrfunc_ud, list_of_binContents_2pointcorrfunc_ud)
avg_bin_content_rep, avg_bin_edges_rep = Average2PointCorrFunction(list_of_binEdges_2pointcorrfunc_rep, list_of_binContents_2pointcorrfunc_rep)

#measures the incompatibility between TS distributions with TS: number of pairs below "ang_window"
print('-----------------------------------------------')
print('-- TS: Number of pairs separated by less than %i degree --' % ang_window)
print('-----------------------------------------------')

percentile = 0.99
TS_incompatibility = Incompatibility(number_of_pairs_below_ud, number_of_pairs_below_rep, percentile)

#compute the likelihood distribution
log_likelihood_ud = []
log_likelihood_rep = []

#for isotropy
for corrfunc_binContent in list_of_binContents_2pointcorrfunc_ud:

    log_likelihood_ud.append(logLikelihood(avg_bin_content_ud, avg_bin_edges_ud, corrfunc_binContent, ang_window))
    print(len(log_likelihood_ud),' samples done!')

#for samples with repeaters
for corrfunc_binContent in list_of_binContents_2pointcorrfunc_rep:

    log_likelihood_rep.append(logLikelihood(avg_bin_content_ud, avg_bin_edges_ud, corrfunc_binContent, ang_window))
    print(len(log_likelihood_rep),' samples done!')

#--------------------------------
# draw 2 point correlation function
#--------------------------------
fig_2pcf = plt.figure(figsize=(10,8)) #creates figure
ax_2pcf = fig_2pcf.add_subplot(111) #create subplot with a set of axis with

ax_2pcf.plot(avg_bin_edges_ud, avg_bin_content_ud, label='Isotropy')
ax_2pcf.plot(avg_bin_edges_rep, avg_bin_content_rep, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))
#ax_2pcf.hist(N_doublets_below_list_rep, bins = max(N_doublets_below_list_rep) - min(N_doublets_below_list_rep), range=[min(N_doublets_below_list_rep), max(N_doublets_below_list_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ hour' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

ax_2pcf.set_title(r'$\Psi$ distribution', fontsize=24)
ax_2pcf.set_xlabel(r'$\Psi (^\circ)$', fontsize=20)
ax_2pcf.set_ylabel(r'Number of pairs', fontsize=20)
ax_2pcf.tick_params(axis='both', which='major', labelsize=20)
ax_2pcf.legend(loc='upper left', fontsize=18)
ax_2pcf.set_yscale('log')
ax_2pcf.set_ylim(1, 1e4)

fig_2pcf.savefig('./results/2PointCorrFunction_%s_TotalEvents_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, N_EVENTS, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#--------------------------------
# draw figure with the estimator
#--------------------------------
fig_est = plt.figure(figsize=(10,8)) #create figure
ax_est = fig_est.add_subplot(111) #create subplot with a set of axis with

ax_est.hist(number_of_pairs_below_ud, bins = 100 , range=[min(number_of_pairs_below_ud), max(number_of_pairs_below_ud)], alpha=0.5, label='Isotropy')
ax_est.hist(number_of_pairs_below_rep, bins = 100 , range=[min(number_of_pairs_below_rep), max(number_of_pairs_below_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))
#ax_est.hist(N_doublets_below_list_rep, bins = max(N_doublets_below_list_rep) - min(N_doublets_below_list_rep), range=[min(N_doublets_below_list_rep), max(N_doublets_below_list_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ hour' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#Fit the distribution of estimators for the UD and Rep distributions
# print('\n ###### FIT PARAMETERS #######\n')
# print('UD distribution:')
# x_fit_ud, y_fit_ud, parameters_ud, parameter_error_ud, covariance_ud = FitEstimatorDist(content_ud, bins_ud, N_doublets_below_list_ud)
#
# print('\nRepeater distribution:\n')
# x_fit_rep, y_fit_rep, parameters_rep, parameter_error_rep, covariance_rep = FitEstimatorDist(content_rep, bins_rep, N_doublets_below_list_rep)
#
# #plot the fits to the distribution
# ax_est.plot(x_fit_ud, y_fit_ud, color='tab:blue', linewidth=2)
# ax_est.plot(x_fit_rep, y_fit_rep, color='darkorange', linewidth = 2)

ax_est.set_title(r'$N_{%.0f^\circ}$-distribution' % (ang_window), fontsize=24)
ax_est.set_xlabel(r'$N_{%.0f^\circ}$' % (ang_window), fontsize=20)
ax_est.set_ylabel(r'Arb. units', fontsize=20)
ax_est.tick_params(axis='both', which='major', labelsize=20)
ax_est.legend(loc='upper right', fontsize=18)
ax_est.set_ylim(0, 50)
#ax_est.text(624500,35, TS_incompatibility, ha='center', va='bottom', fontsize=16, linespacing=1.5, wrap=True, bbox=dict(facecolor='grey', alpha=0.2))
fig_est.savefig('./results/NumberOfPairsBelow_%i_Distribution_%s_TotalEvents_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (ang_window, REP_DATE, N_EVENTS, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#----------------------------------
# draw figure with likelihood
#----------------------------------
print('----------------------------------------------------------')
print('-- TS: log Likelihood  for angular scales < %.0f degree --' % ang_window)
print('----------------------------------------------------------')
TS_incompatibility = Incompatibility(log_likelihood_ud, log_likelihood_rep, percentile)

fig_like = plt.figure(figsize=(10,8)) #create figure
ax_like = fig_like.add_subplot(111) #create subplot with a set of axis with

ax_like.hist(log_likelihood_ud, bins = 100 , range=[min(log_likelihood_ud), max(log_likelihood_ud)], alpha=0.5, label='Isotropy')
ax_like.hist(log_likelihood_rep, bins = 100 , range=[min(log_likelihood_rep), max(log_likelihood_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#ax_like.hist(N_doublets_below_list_rep, bins = max(N_doublets_below_list_rep) - min(N_doublets_below_list_rep), range=[min(N_doublets_below_list_rep), max(N_doublets_below_list_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ hour' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#Fit the distribution of estimators for the UD and Rep distributions
# print('\n ###### FIT PARAMETERS #######\n')
# print('UD distribution:')
# x_fit_ud, y_fit_ud, parameters_ud, parameter_error_ud, covariance_ud = FitEstimatorDist(content_ud, bins_ud, N_doublets_below_list_ud)
#
# print('\nRepeater distribution:\n')
# x_fit_rep, y_fit_rep, parameters_rep, parameter_error_rep, covariance_rep = FitEstimatorDist(content_rep, bins_rep, N_doublets_below_list_rep)
#
# #plot the fits to the distribution
# ax_like.plot(x_fit_ud, y_fit_ud, color='tab:blue', linewidth=2)
# ax_like.plot(x_fit_rep, y_fit_rep, color='darkorange', linewidth = 2)

ax_like.set_title(r'$\ln \mathcal{L}_{%.0f^\circ}$-distribution' % (ang_window), fontsize=24)
ax_like.set_xlabel(r'$\ln \mathcal{L}_{%.0f^\circ}$' % (ang_window), fontsize=20)
ax_like.set_ylabel(r'Arb. units', fontsize=20)
ax_like.tick_params(axis='both', which='major', labelsize=20)
ax_like.legend(loc='upper right', fontsize=18)
#ax_like.text(-7.755e6, 35, TS_incompatibility, ha='center', va='bottom', fontsize=16, linespacing=1.5, wrap=True, bbox=dict(facecolor='grey', alpha=0.2))
ax_like.set_ylim(0, 50)

fig_like.savefig('./results/LogLikelihoodBelow_%i_Distribution_%s_TotalEvents_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (ang_window, REP_DATE, N_EVENTS, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))
