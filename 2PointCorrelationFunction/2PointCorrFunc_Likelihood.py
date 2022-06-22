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
def Read2PointCorrFunc(filename, max_sep):

    corrfunc = pd.read_parquet(filename, engine='fastparquet')

    corrfunc = corrfunc[corrfunc['2pointcorr_bin_edges'] <= max_sep]

    print(corrfunc.head())

    bin_edges = corrfunc['2pointcorr_bin_edges'].to_numpy()
    bin_contents = corrfunc['2pointcorr_bin_content'].to_numpy()

    return bin_edges, bin_contents

#define the log likelihood function
def logLikelihood(avg_bin_content, obs_bin_content):

    log_obs_factorial = []
    factorial_sum = []

    for obs in obs_bin_content:
        for j in range(1, int(obs)+1):
            log_obs_factorial.append(math.log(j))

        factorial_sum.append(sum(log_obs_factorial))

    return -np.sum(avg_bin_content) + np.sum(obs_bin_content*np.log(avg_bin_content)) - sum(factorial_sum)

#defines the incompatibility between distributions
def Incompatibility(likelihood_list_ud, likelihood_list_rep, percentile):

    #computes the mean and std of the distribution of likelihood for Isotropic dist.
    mean_ud = np.mean(estimator_list_ud)
    sigma_ud = math.sqrt(np.var(estimator_list_ud))

    quantile = np.quantile(estimator_list_rep, percentile)

    print('Mean of test statistic for UD distribution', mean_ud)
    print('RMS of test statistic for UD distribution', sigma_ud)
    print('The q(',percentile,') for the distribution of TS for Repeater samples is', quantile)

    return (quantile - mean_ud)/sigma_ud

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
#set path to dir with uniform dist files
path_of_uniform_dir = '../DataSets/Vertical/UD_large_stats/2PointCorrFunc/'

#list to hold histograms 2point correlation functions
list_of_hist_binedges_2pointcorrfunc = []
list_of_hist_contents_2pointcorrfunc = []

#sets maximum angular separation for 2-point corr. func. in degrees!
max_sep = 5 #degree
ang_window = 1 #degree

#initializes the number of pairs below
number_of_pairs_below = []

for file in os.listdir(path_of_uniform_dir):

    filename = os.path.join(path_of_uniform_dir,file)

    if os.path.isfile(filename) and '2PointCorrelationFunction_N_' in filename and len(number_of_pairs_below) < 10:

        begin_task = datetime.now()

        bin_edges, bin_contents = Read2PointCorrFunc(filename, max_sep)

        list_of_hist_binedges_2pointcorrfunc.append(bin_edges)
        list_of_hist_contents_2pointcorrfunc.append(bin_contents)

        number_of_pairs_scalar = sum([bin_contents[i] for i in range(len(bin_contents)) if bin_edges[i] <= ang_window])
        number_of_pairs_below.append(int(number_of_pairs_scalar))

        print('This took', datetime.now() - begin_task)

avg_bin_content, avg_bin_edges = Average2PointCorrFunction(list_of_hist_binedges_2pointcorrfunc, list_of_hist_contents_2pointcorrfunc)

print(min(avg_bin_edges), max(avg_bin_edges))

#compute the likelihood distribution
log_likelihood = []

for corr_func_bin_content in list_of_hist_contents_2pointcorrfunc:

    log_likelihood.append(logLikelihood(avg_bin_content, corr_func_bin_content))
    print(len(log_likelihood),' samples done!')

#--------------------------------
# draw 2 point correlation function
#--------------------------------
fig_2pcf = plt.figure(figsize=(10,8)) #creates figure
ax_2pcf = fig_2pcf.add_subplot(111) #create subplot with a set of axis with

ax_2pcf.plot(avg_bin_edges, avg_bin_content, label='Isotropy')
#ax_2pcf.hist(N_doublets_below_list_rep, bins = max(N_doublets_below_list_rep) - min(N_doublets_below_list_rep), range=[min(N_doublets_below_list_rep), max(N_doublets_below_list_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ hour' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

ax_2pcf.set_title(r'$\Psi$ distribution', fontsize=24)
ax_2pcf.set_xlabel(r'$\Psi (^\circ)$', fontsize=20)
ax_2pcf.set_ylabel(r'Number of pairs', fontsize=20)
ax_2pcf.tick_params(axis='both', which='major', labelsize=20)
ax_2pcf.legend(loc='upper left', fontsize=18)
#ax_2pcf.set_ylim(0, 50)

fig_2pcf.savefig('./results/2PointCorrFunction.pdf') #Number_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#--------------------------------
# draw figure with the estimator
#--------------------------------
fig_est = plt.figure(figsize=(10,8)) #create figure
ax_est = fig_est.add_subplot(111) #create subplot with a set of axis with

ax_est.hist(number_of_pairs_below, bins = 100 , range=[min(number_of_pairs_below), max(number_of_pairs_below)], alpha=0.5, label='Isotropy')
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

ax_est.set_title(r'$\hat{N}(\Psi < %.0f^\circ$) distribution' % (ang_window), fontsize=24)
ax_est.set_xlabel(r'$\hat{N}(\Psi < %.0f^\circ$)' % (ang_window), fontsize=20)
ax_est.set_ylabel(r'Arb. units', fontsize=20)
ax_est.tick_params(axis='both', which='major', labelsize=20)
ax_est.legend(loc='upper right', fontsize=18)
#ax_est.set_ylim(0, 50)

fig_est.savefig('./results/NumberOfPairsBelow_%i_Distribution.pdf' % ang_window) #)distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#compute the 2 point correlation function for the file with a point repeater
# path_to_rep_file = '../DataSets/Vertical/MockData_Repeaters/Repeater_FixedPosAndDate_large_stats/'
# repeater_file = 'ExpRepeater_Date_2015-01-01T00:00:00_Period_86164_TotalEvents_100000_AcceptedRepEvents_100_3834357_1.parquet'
#
# #defines the maximum angular seperation
# max_ang_sep = math.pi/4
#
# rep_2pointcorrfunc = Get2PointCorrFunc(path_to_rep_file + repeater_file,'rep_ud_ra', 'rep_ud_dec', 'rep_ud_gpstime', max_ang_sep, 10000)
#
# #print(len(rep_2pointcorrfunc_2))
#
# plt.hist(rep_2pointcorrfunc, bins = 90, range = [0,math.degrees(max_ang_sep)], label='Point repeater')
# plt.yscale('log')
#
# #plt.hist(log_likelihood, bins = 10, range = [min(log_likelihood), max(log_likelihood)])
#
# plt.legend()
# plt.show()

#hist_content, hist_bin_edges = np.histogram(list_of_ang_dist, 180)

#plt.plot(hist_bin_edges[1:], hist_content, color='darkorange')

#plt.show()

#make average 2 point correlation function
# hist = ROOT.TH1D('hist','hist', 180, 0, 180)
#
# for angle in list_of_ang_dist:
#     hist.Fill(angle)
#
# hist_list = []
#
# hist_list.append(hist)
#
# TwoPointAverageCorrFunc = ROOT.TProfile('TwoPointAverageCorrFunc','TwoPointAverageCorrFunc', 180, 0, 180)

#for subdir, dirs, files in os.walk(path_of_dir):
# for filename in os.listdir(path_of_dir):
#
#     f = os.path.join(path_of_dir,filename)
#
#     if os.path.isfile(f) and 'Ud_events_with_tau_' in f: # and len(list_of_ordered_taus) < 10:
#
#         df = pd.read_parquet(f, engine='fastparquet')
#
#         tau_ud = np.multiply(df["tau (s)"].to_numpy(),1/86164)
#         list_of_ordered_taus.append(sorted(tau_ud))
#
#
# #read file with tau values for Auger data
# df_repeater = pd.read_parquet('/home/miguel/Documents/Repeaters/repeaters_analysis_mockdata/output/Rep_events_with_tau_Date_2015-01-01T00:00:00_Period_3600.0_TotalEvents_100000_AcceptedRepEvents_67.parquet', engine='fastparquet')
# N_ACCEPTED_REP_EVENTS = '67'
#
# #store tau values in sidereal days
# tau_repeater = np.multiply(df_repeater["tau (s)"].to_numpy(),1/86164)
#
#
# #---------------------------------------
# # To plot histograms of tau distribution
# #---------------------------------------
#
# #list to save all tau values from all realizations of isotropic distribution
# tau_ud_all = []
#
# for tau_list in list_of_ordered_taus:
#     for tau in tau_list:
#         tau_ud_all.append(tau)
#
# #perform KS test to verify the compatibity between tau dists with and without repeater
# ks_stat_value, ks_p_value = stats.kstest(tau_repeater, tau_ud_all)
#
# #plot log10 of tau distributions
# fig_tau_log = plt.figure(figsize=(10,8)) #create figure
# ax_tau_log = fig_tau_log.add_subplot(111) #create subplot with a set of axis with
#
# ax_tau_log.hist(np.log10(tau_repeater),bins=200, density=True, range=[-5,4], alpha = 0.5, label=r'Isotropy + ' + N_ACCEPTED_REP_EVENTS + r' events from repeater w/ $\tau = 1$ hour')
# ax_tau_log.hist(np.log10(tau_ud_all),bins=200, density=True, range=[-5,4], alpha = 0.5, label=r'Isotropy')
# ax_tau_log.plot([],[],lw=0,label=r'KS test: $p$-value = {%.3f}' % ks_p_value)
# ax_tau_log.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Delta \theta = 1^\circ$', fontsize=24)
# ax_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{sidereal days})$', fontsize=20)
# ax_tau_log.set_ylabel(r'Prob. density', fontsize=20)
# ax_tau_log.tick_params(axis='both', which='major', labelsize=20)
# ax_tau_log.set_yscale('log')
# ax_tau_log.legend(loc='best', fontsize=20)
#
# fig_tau_log.savefig('./results/Average_log10tau_distribution_wRepeater.pdf')
#
# fig_tau = plt.figure(figsize=(10,8)) #create figure
# ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with
#
# ax_tau.hist(tau_repeater,bins=200, density=True, range=[0,10], alpha = 0.5, label=r'Isotropy + ' + N_ACCEPTED_REP_EVENTS + r' events from repeater w/ $\tau = 1$ hour')
# ax_tau.hist(tau_ud_all,bins=200, density=True, range=[0,10], alpha = 0.5, label=r'Isotropy')
#
# ax_tau.set_title(r'$\tau$ distribution for angular window $\Delta \theta = 1^\circ$',fontsize=24)
# ax_tau.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
# ax_tau.set_ylabel(r'Prob. density', fontsize=20)
# ax_tau.tick_params(axis='both', which='major', labelsize=20)
# ax_tau.set_xlim([0,10])
# ax_tau.set_yscale('log')
# ax_tau.legend(loc='best', fontsize=20)
#
# fig_tau.savefig('./results/Average_tau_distribution_wRepeater.pdf')
#
# #list with the integration limits
# list_of_integration_lims = np.arange(0,2,1)
#
# #list of p_values for each integration range
# list_of_p_values = []
#
# for i in range(1,len(list_of_integration_lims)):
#
#     lower_lim = list_of_integration_lims[i-1]
#     upper_lim = list_of_integration_lims[i]
#
#     #estimator value for repeater mock data set
#     repeater_estimator = len( [tau for tau in tau_repeater if (tau > lower_lim and tau < upper_lim)])
#
#     #estimator distribution for the many isotropic realizations
#     estimator_list = []
#
#     for tau_list in list_of_ordered_taus:
#         estimator = len( [tau for tau in tau_list if (tau > lower_lim and tau < upper_lim)] )
#         estimator_list.append(estimator)
#
#     #compute the number of RMS between average and mockdata set point
#     print('Estimator for Mock data set is', abs(np.mean(estimator_list) - repeater_estimator)/math.sqrt(np.var(estimator_list)),'sigma away from estimator dist mean')
#
#     #integral above the auger datum point
#     above_rep_data = len([est for est in estimator_list if est >= repeater_estimator])
#     p_value = above_rep_data/len(estimator_list)
#
#     print('p-value for tau distribution estimator', p_value)
#
#     list_of_p_values.append(p_value)
#
#     #draw figure with the estimator
#     fig_est = plt.figure(figsize=(10,8)) #create figure
#     ax_est = fig_est.add_subplot(111) #create subplot with a set of axis with
#
#     content, bins, _ = ax_est.hist(estimator_list, bins = max(estimator_list), range=[min(estimator_list), max(estimator_list)], density=True, label='Uniform distribution')
#     ax_est.axvline(repeater_estimator, 0, max(content), linestyle = 'dashed', color = 'darkorange', label=r'Repeater data')
#     ax_est.plot([],linewidth=0, label=r'$\hat{I} = \displaystyle\int_{%i}^{%i} \displaystyle\frac{\textrm{d} N}{\textrm{d} \tau} \textrm{d}\tau$' % (lower_lim,upper_lim))
#     ax_est.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (p_value))
#
#     ax_est.set_title(r'$\hat{I}$ distribution', fontsize=24)
#     ax_est.set_xlabel(r'$\hat{I}$', fontsize=20)
#     ax_est.set_ylabel(r'Prob. density', fontsize=20)
#     ax_est.tick_params(axis='both', which='major', labelsize=20)
#     ax_est.legend(loc='best', fontsize=20)
#
#     fig_est.savefig('./results/Estimator_distribution_histogram.pdf')


# #draw auger and ALL uniform dist data in linear scale
# fig_tau = plt.figure(figsize=(10,8)) #create figure
# ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with
#
# ax_tau.hist(tau_auger,bins=200, density=True, range=[0,6000], alpha = 0.5, label=r'Auger data ($N =$ ' + str(N_auger_events) + ')')
# ax_tau.hist(tau_ud_all,bins=200, density=True, range=[0,6000], alpha = 0.5, label=r'Uniform dist ($N =$ ' + str(file_counter) + r' $\times$ ' + str(N_auger_events) + ')')
#
# ax_tau.set_title(r'$\tau$ distribution for angular window $\Delta \theta =$ ' + str(round(math.degrees(selection_info.iloc[0,3]),2)) + '$^{\circ}$',fontsize=24)
# ax_tau.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
# ax_tau.set_ylabel(r'Prob. density', fontsize=20)
# ax_tau.tick_params(axis='both', which='major', labelsize=20)
# ax_tau.set_xlim([0,6000])
# ax_tau.legend(loc='best', fontsize=20)
#
# #for the inset
# ax_tau_inset = inset_axes(ax_tau, width='100%', height='100%', bbox_to_anchor=(.5,.25,.4,.5), bbox_transform=ax_tau.transAxes, loc=3)
# ax_tau_inset.hist(tau_auger,bins=100, density=True, range=[0,10], alpha = 0.5)
# ax_tau_inset.hist(tau_ud_all,bins=100, density=True, range=[0,10], alpha = 0.5)
#
# ax_tau_inset.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
# ax_tau_inset.set_ylabel(r'Arb. Units', fontsize=20)
# ax_tau_inset.tick_params(axis='both', which='major', labelsize=20)
# ax_tau_inset.set_xlim([0,10])
#
# fig_tau.savefig('Average_tau_distribution_histogram.pdf')
#
#
#
#
#
# #compute the p-value
# above_data_integral = 0
#
# for i in range(len(bin_content)):
#     if( bins[i] > realization):
#         above_data_integral+=bin_content[i]
#
# #integral of dist of estimator
# integral = sum(bin_content)
#
# print('The p-value for the null hypothesis is',above_data_integral/integral)
#
#
# #--------------------------------------------------
# # Compute the comulative function for each histogram
# #--------------------------------------------------
# pdf_tau_auger, bins_tau_auger = np.histogram(tau_auger, bins=200)
# pdf_tau_ud_all, bins_tau_ud_all = np.histogram(tau_ud_all, bins=200)
#
# pdf_logtau_auger, bins_logtau_auger = np.histogram(np.log10(tau_auger), bins=200)
# pdf_logtau_ud_all, bins_logtau_ud_all = np.histogram(np.log10(tau_ud_all), bins=200)
#
# #make histograms with root just to check if everything is working
# hist_tau_auger = ROOT.TH1D('hist_tau_auger','hist_tau_auger',200, min(tau_ud_all), max(tau_ud_all))
# hist_tau_ud_all = ROOT.TH1D('hist_tau_ud_all','hist_tau_ud_all',200, min(tau_ud_all), max(tau_ud_all))
#
# #make tprofile of tau_ud_all
# profile_tau_ud_all = ROOT.TProfile('profile_tau_ud_all','profile_tau_ud_all',200, min(np.log10(tau_ud_all)), max(np.log10(tau_ud_all)))
#
# for t in tau_auger:
#     hist_tau_auger.Fill(t)
#
# for t in tau_ud_all:
#     hist_tau_ud_all.Fill(t)
#
# for t in np.log10(tau_ud_all):
#     profile_tau_ud_all.Fill(t,1)
#
# profile_content_tau_ud_all = []
# profile_bins_tau_ud_all = []
#
# for i in range(1,profile_tau_ud_all.GetNbinsX()):
#     profile_content_tau_ud_all.append(profile_tau_ud_all.GetBinContent(i))
#     profile_bins_tau_ud_all.append(profile_tau_ud_all.GetBinCenter(i))
#
# hist_cdf_tau_auger = hist_tau_auger.GetCumulative()
# hist_cdf_tau_ud_all = hist_tau_ud_all.GetCumulative()
#
# cdf_root_tau_auger = []
# bins_root_tau_auger = []
#
# cdf_root_tau_ud_all = []
# bins_root_tau_ud_all = []
#
# for i in range(1,hist_cdf_tau_auger.GetNbinsX()):
#     cdf_root_tau_auger.append(hist_cdf_tau_auger.GetBinContent(i))
#     bins_root_tau_auger.append(hist_cdf_tau_auger.GetBinCenter(i))
#
# for i in range(1,hist_cdf_tau_ud_all.GetNbinsX()):
#     cdf_root_tau_ud_all.append(hist_cdf_tau_ud_all.GetBinContent(i))
#     bins_root_tau_ud_all.append(hist_cdf_tau_ud_all.GetBinCenter(i))
#
# #make CDFs with numpy
# cdf_tau_auger = np.cumsum(pdf_tau_auger)/sum(pdf_tau_auger)
# cdf_tau_ud_all = np.cumsum(pdf_tau_ud_all)/sum(pdf_tau_ud_all)
#
# cdf_logtau_auger = np.cumsum(pdf_logtau_auger)/sum(pdf_tau_auger)
# cdf_logtau_ud_all = np.cumsum(pdf_logtau_ud_all)/sum(pdf_tau_ud_all)
#
# diff_between_CDFs = cdf_tau_auger - cdf_tau_ud_all
# diff_between_CDFs_log = cdf_logtau_auger - cdf_logtau_ud_all
#
# print('ROOT: KS test max diff', hist_tau_auger.KolmogorovTest(hist_tau_ud_all, 'M'))
#
# print('Miguel: Maximum difference between CDFs of tau  = ', max(diff_between_CDFs))
# print('Miguel: Maximum difference between CDFs of log tau  = ', max(diff_between_CDFs_log))
#
# #draw figure of CDF of tau
# fig_cdf = plt.figure(figsize=(10,8)) #create figure
# ax_cdf = fig_cdf.add_subplot(111) #create subplot with a set of axis with
#
# #data, bins, patches = ax_tau.hist(tau_auger,bins=200, density=True, range=[-1.5,5], alpha = 0.5, label=r'Auger data ($N =$ ' + str(N_auger_events) + ')')
# ax_cdf.plot(bins_tau_auger[1:], cdf_tau_auger, linewidth=2, label=r'Auger Data')
# #ax_cdf.plot(bins_root_tau_auger[0:], np.multiply(cdf_root_tau_auger,1/len(tau_auger)), linewidth=2, label=r'Auger Data with Root')
# ax_cdf.plot(bins_tau_ud_all[1:], cdf_tau_ud_all, linewidth=2, label=r'Uniform distribution')
# #ax_cdf.plot(bins_root_tau_ud_all[0:], np.multiply(cdf_root_tau_ud_all,1/len(tau_ud_all)), linewidth=2, label=r'Auger Data with Root')
# ax_cdf.set_title(r'$N(\tau)$', fontsize=24)
# ax_cdf.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
# ax_cdf.set_ylabel(r'$\displaystyle\frac{N(\tau)}{N_{\textrm{evt}}}$', fontsize=20)
# ax_cdf.tick_params(axis='both', which='major', labelsize=20)
# ax_cdf.set_xlim([0,6000])
# #ax_tau.set_yscale('log')
# ax_cdf.legend(loc='best', fontsize=20)
#
# fig_cdf.savefig('CDF_tau_histogram.pdf')
#
#
# #draw figure of CDF of log10(tau)
# fig_logcdf = plt.figure(figsize=(10,8)) #create figure
# #fig_logcdf.subplot_adjust(right=0.9)
# ax_logcdf = fig_logcdf.add_subplot(111) #create subplot with a set of axis with
#
# #data, bins, patches = ax_tau.hist(tau_auger,bins=200, density=True, range=[-1.5,5], alpha = 0.5, label=r'Auger data ($N =$ ' + str(N_auger_events) + ')')
# ax_logcdf.plot(bins_logtau_auger[1:], cdf_logtau_auger, linewidth=2, label=r'Auger Data')
# ax_logcdf.plot(bins_logtau_ud_all[1:], cdf_logtau_ud_all, linewidth=2, label=r'Uniform distribution')
# ax_logcdf.set_title(r'$N(\log_{10}\tau)$', fontsize=24)
# ax_logcdf.set_xlabel(r'$\log_{10} (\tau /\textrm{sidereal day})$', fontsize=20)
# ax_logcdf.set_ylabel(r'$\displaystyle\frac{N(\log_{10}\tau)}{N_{\textrm{evt}}}$', fontsize=20)
# ax_logcdf.tick_params(axis='both', which='major', labelsize=20)
# ax_logcdf.set_xlim([-2,4])
# ax_logcdf.set_yscale('log')
# ax_logcdf.legend(loc='best', fontsize=20)
#
# fig_logcdf.savefig('CDF_log10_tau_histogram.pdf')

#print profile of tau dis root
fig_profile = plt.figure(figsize=(10,8)) #create figure
ax_profile = fig_profile.add_subplot(111) #create subplot with a set of axis with

#print(profile_bins_tau_ud_all)
#print(profile_content_tau_ud_all)

#data, bins, patches = ax_tau.hist(tau_auger,bins=200, density=True, range=[-1.5,5], alpha = 0.5, label=r'Auger data ($N =$ ' + str(N_auger_events) + ')')
#ax_profile.plot(profile_bins_tau_ud_all, profile_content_tau_ud_all, linewidth=2, label=r'Auger Data')

# ax_logcdf.plot(bins_logtau_ud_all[1:], cdf_logtau_ud_all, linewidth=2, label=r'Uniform distribution')
# ax_logcdf.set_title(r'CDF of $\displaystyle\frac{dN}{d\log_{10} \tau}$', fontsize=24)
# ax_logcdf.set_xlabel(r'$\log_{10} (\tau /sidereal days)$', fontsize=20)
# ax_logcdf.set_ylabel(r'CDF', fontsize=20)
# ax_logcdf.tick_params(axis='both', which='major', labelsize=20)
# ax_logcdf.set_xlim([-2,4])
# ax_logcdf.set_yscale('log')
# ax_logcdf.legend(loc='best', fontsize=20)

#fig_profile.savefig('Average_log10_tau_profile.pdf')
