import math
import numpy as np

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#for data manipulation
import pandas as pd

#for manipulation and conversion between times
from astropy.time import Time

#to read files
import os

#import ROOT

#for statistics
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import chisquare

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)
plt.rcParams["figure.autolayout"] = True

#define the gaussian function
def Gaussian(x, mu, sigma, norm, shift):

    return shift + (norm/math.sqrt(2*math.pi)*sigma)*np.exp(-(x - mu) ** 2 / (2* sigma ** 2))

#define the average distribution of tau for many realizations
def AverageTauDist(list_of_tau_arrays, nbins, hist_min, hist_max):

    list_of_bin_edges = []
    list_of_bin_contents = []

    for tau_array in list_of_tau_arrays:

        bin_contents, bin_edges = np.histogram(tau_array, nbins, range=[hist_min,hist_max])

        list_of_bin_edges.append(bin_edges)
        list_of_bin_contents.append(bin_contents)

    avg_bin_edges = list_of_bin_edges[0][1:]
    avg_bin_content = []

    for i in range(nbins):

        bin_contents = []

        for hist_content in list_of_bin_contents:

            bin_contents.append(hist_content[i])

        avg_bin_content.append(np.mean(bin_contents))

    return avg_bin_edges, avg_bin_content

#to get list of ordered taus and list with all taus from files
def FromFiles_to_TauDist(path_to_dir, name_of_files):

    list_of_ordered_tau_arrays = []
    tau_ud_all = []

    #to count the number of files read
    file_counter = 0

    #for subdir, dirs, files in os.walk(path_of_dir):
    for filename in os.listdir(path_to_dir):

        f = os.path.join(path_to_dir,filename)

        if os.path.isfile(f) and name_of_files in f:

            df = pd.read_parquet(f, engine='fastparquet')

            tau_ud = np.divide(df["tau (s)"].to_numpy(),86164)

            #list_of_log_tau_arrays.append(np.log10(tau_ud))

            list_of_ordered_tau_arrays.append(sorted(tau_ud))

            for tau in tau_ud:
                tau_ud_all.append(tau)

            file_counter+=1

            print(file_counter,'files read!')

    return tau_ud_all, list_of_ordered_tau_arrays


#computes the distribution of the estimator
def EstimatorDist(list_of_ordered_taus, lower_lim, upper_lim):

    #estimator distribution for the many isotropic realizations
    estimator_list = []

    for tau_list in list_of_ordered_taus:
        estimator = len( [tau for tau in tau_list if (tau > lower_lim and tau < upper_lim)] )
        estimator_list.append(estimator)

    return estimator_list

#compute chi^2 for a fit
def Chi_Square(y_data, y_true, sigma):

    r = y_true - y_data

    return sum(( r / sigma)*( r / sigma))

#compute the comulative distribution function
def ComulativeDistFunc(data, nbins):

    pdf_data, pdf_bin_edges = np.histogram(data, nbins)

    cdf = np.cumsum(pdf_data)/sum(pdf_data)

    return pdf_bin_edges[1:], cdf

#draws an exponential envelop
def LogExpEnvelop(x, rate):

    return rate*np.exp(-rate*np.exp(x) + x)

#set path to dir with uniform dist files
path_to_dir_ud = '/home/miguel/Documents/Repeaters_Analysis/DataSets/Inclined+Vertical/UD_AugerOpenData_stats'
#path_to_dir_rep = '/home/miguelm/Documents/Anisotropies/Repeater_Analysis/DataSets/MockData_Repeaters/Rep_large_stats/'

#define tau threshold in log10(tau) with tau in days
tau_th = 0

#list to hold all tau values from all data sets of isotropy
tau_ud_all, list_of_ordered_taus_ud = FromFiles_to_TauDist(path_to_dir_ud, 'UD_AllAugerPubEvents_with_tau')

#tau_rep_all, list_of_ordered_taus_rep = FromFiles_to_TauDist(path_to_dir_rep, 'Rep_events_with_tau')

list_of_log_tau_arrays = [np.log10(tau) for tau in list_of_ordered_taus_ud]

#read file with tau values for repeater data
# path_to_repeaters = '/home/miguelm/Documents/Anisotropies/Repeater_Analysis/DataSets/MockData_Repeaters/'
# PERIOD_OF_REP = '3600'
# N_ACCEPTED_REP_EVENTS = '100'
# N_INTENSITY = '5'
#
# df_repeater = pd.read_parquet(path_to_repeaters + 'Rep_events_with_tau_Period_' + PERIOD_OF_REP + '_TotalEvents_100000_AcceptedRepEvents_' + N_ACCEPTED_REP_EVENTS + '_MaxRepIntensity_' + N_INTENSITY + '.parquet', engine='fastparquet')
#
# #store tau values in sidereal days
# tau_repeater = np.divide(df_repeater["tau (s)"].to_numpy(),86164)

#----------------------
# store the tau distribution from auger data and saves important info from the selection info file
#----------------------
path_to_auger_data = '../results/'
auger_data_file = 'AugerOpenData_AllEvents_with_tau.parquet'
auger_selection_file = 'AugerOpenData_AllEvents_SelectionInfo.parquet'

auger_data = pd.read_parquet(path_to_auger_data + auger_data_file, engine='fastparquet')
auger_selection_info = pd.read_parquet(path_to_auger_data + auger_selection_file, engine='fastparquet')

#store tau values in sidereal days
tau_auger = np.divide(auger_data["tau (s)"].to_numpy(), 86164)

#stores important info from auger selection info file
N_events = int(auger_selection_info.iloc[0]['N_events'])
theta_min = float(auger_selection_info.iloc[0]['Theta_min'])
theta_max = float(auger_selection_info.iloc[0]['Theta_max'])
ang_window = math.degrees(float(auger_selection_info.iloc[0]['Ang_window']))
t_begin = Time(auger_selection_info.iloc[0]['t_begin'], format='fits').gps
t_end = Time(auger_selection_info.iloc[0]['t_end'], format = 'fits').gps

#computes the average rate of events in events per sidereal day
average_rate = 86164 * N_events/(t_end - t_begin)

print('Average rate', average_rate)
#---------------------------------------
# To plot histograms of tau distribution
#---------------------------------------

#perform KS test to verify the compatibity between tau dists with and without repeater
ks_stat_value, ks_p_value = stats.kstest(tau_auger, tau_ud_all)

#plot log10 of tau distributions
fig_tau_log = plt.figure(figsize=(10,8)) #create figure
ax_tau_log = fig_tau_log.add_subplot(111) #create subplot with a set of axis with

ax_tau_log.hist(np.log10(tau_auger), bins=200, range=[-2,4], alpha = 0.5, label=r'Auger Data: $N_{\textrm{evt}} = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))

log_tau_avg_hist_edges, log_tau_avg_hist_content = AverageTauDist(list_of_log_tau_arrays, 200, -2, 4)

ax_tau_log.plot(log_tau_avg_hist_edges, log_tau_avg_hist_content, label=r'Isotropy')
#ax_tau_log.plot(np.arange(-2,4,0.01), len(tau_auger)*LogExpEnvelop(np.arange(-2,4,0.01), average_rate), color = 'purple', linestyle='--', label=r'Exponential Envelop',)

ax_tau_log.plot([],[],lw=0,label=r'KS test: $p$-value = {%.10f}' % ks_p_value)

ax_tau_log.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Delta \theta = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{sidereal days})$', fontsize=20)
ax_tau_log.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_tau_log.set_yscale('log')
ax_tau_log.set_ylim(1e-2, 1e4)

ax_tau_log.legend(loc='best', fontsize=18)

fig_tau_log.savefig('./AugerOpenData/Average_log10tau_distribution_AugerOpenData.pdf')

#--------------------------------------
# plot of tau distributions
#--------------------------------------
# fig_tau = plt.figure(figsize=(10,8)) #create figure
# ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with
#
# ax_tau.hist(tau_auger,bins=200, range=[0,10], alpha = 0.5, label=r'Isotropy + ' + N_ACCEPTED_REP_EVENTS + r' events from repeater w/ $\tau = 1$ hour')
#
# tau_avg_hist_edges, tau_avg_hist_content = AverageTauDist(list_of_ordered_taus_ud, 200, 0, 10)
#
# ax_tau.plot(tau_avg_hist_edges, tau_avg_hist_content, label=r'Isotropy')
#
# ax_tau.set_title(r'$\tau$ distribution for angular window $\Delta \theta = 1^\circ$',fontsize=24)
# ax_tau.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
# ax_tau.set_ylabel(r'Number of pairs', fontsize=20)
# ax_tau.tick_params(axis='both', which='major', labelsize=20)
# ax_tau.set_xlim([0,10])
# ax_tau.set_yscale('log')
# ax_tau.set_ylim(1e-1,5e2)
#
# ax_tau.legend(loc='upper right', fontsize=18)
#
# fig_tau.savefig('./results/Average_tau_distribution_wRepeater_RandPos_' + PERIOD_OF_REP + '.pdf')
#
#---------------------------------------
# plot of cdf of log tau
#---------------------------------------
fig_cdf_tau_log = plt.figure(figsize=(10,8)) #create figure
ax_cdf_tau_log = fig_cdf_tau_log.add_subplot(111) #create subplot with a set of axis with

cdf_rep_bin_edges, cdf_rep_content = ComulativeDistFunc(np.log10(tau_auger), 100)
cdf_ud_bin_edges, cdf_ud_content = ComulativeDistFunc(np.log10(tau_ud_all), 100)

ax_cdf_tau_log.plot(cdf_rep_bin_edges, cdf_rep_content, label=r'Auger Data: $N_{\textrm{evt}} = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))
ax_cdf_tau_log.plot(cdf_ud_bin_edges, cdf_ud_content, label=r'Isotropy')

ax_cdf_tau_log.set_title(r'$N(\log_{10}(\tau))$ for angular window $\Delta \theta = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_cdf_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{sidereal days})$', fontsize=20)
ax_cdf_tau_log.set_ylabel(r'Arb. Units', fontsize=20)
ax_cdf_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_cdf_tau_log.set_yscale('log')
#ax_cdf_tau_log.set_ylim(1e-2, 1e4)

ax_cdf_tau_log.legend(loc='best', fontsize=18)

fig_cdf_tau_log.savefig('./AugerOpenData/Average_log10tau_CDF_AugerInclinedData.pdf')

#list with the integration limits given in sidereal days!!!!!
list_of_integration_lims = [0,1]

#list of p_values for each integration range
list_of_p_values = []

for i in range(1,len(list_of_integration_lims)):

    lower_lim = list_of_integration_lims[i-1]
    upper_lim = list_of_integration_lims[i]

    #estimator value for repeater mock data set
    auger_data_estimator = len( [tau for tau in tau_auger if (tau > lower_lim and tau < upper_lim)])

    estimator_list = EstimatorDist(list_of_ordered_taus_ud, lower_lim, upper_lim)
    #estimator_list_rep = EstimatorDist(list_of_ordered_taus_rep, lower_lim, upper_lim)

    #compute the number of RMS between average and mockdata set point
    print('Estimator for Mock data set is', abs(np.mean(estimator_list) - auger_data_estimator)/math.sqrt(np.var(estimator_list)),'sigma away from estimator dist mean')

    #integral above the repeater datum point
    above_rep_data = len([est for est in estimator_list if est >= auger_data_estimator])
    p_value = above_rep_data/len(estimator_list)

    print('p-value for tau distribution estimator', p_value)

    list_of_p_values.append(p_value)

    #draw figure with the estimator
    fig_est = plt.figure(figsize=(10,8)) #create figure
    ax_est = fig_est.add_subplot(111) #create subplot with a set of axis with

    content, bins, _ = ax_est.hist(estimator_list, bins = max(estimator_list) - min(estimator_list), range=[min(estimator_list), max(estimator_list)], alpha=0.5, label='Uniform distribution')
    #content_rep, bins_rep, _ = ax_est.hist(estimator_list_rep, bins = max(estimator_list_rep) - min(estimator_list_rep), range=[min(estimator_list_rep), max(estimator_list_rep)], alpha=0.5, label='Repeater distribution')

    ax_est.axvline(auger_data_estimator, 0, max(content), linestyle = 'dashed', color = 'darkorange', label=r'Auger data')

    #fit the distribution of the estimator
    #sigma_data = np.sqrt(content)

    #print(sigma_data)

    # init_parameters = [np.mean(estimator_list), math.sqrt(np.var(estimator_list)), 1, 0]
    #
    # parameters, covariance = curve_fit(Gaussian, np.array(bins[1:]), np.array(content), p0=init_parameters) # sigma = sigma_data, absolute_sigma=True)
    #
    # x_gauss_fit = np.arange(min(estimator_list), max(estimator_list), 0.01)
    # y_gauss_fit = Gaussian(x_gauss_fit, *parameters)
    #
    # #expected value for each bin
    # expected_bin_content = []
    #
    # for i in range(len(bins)-1):
    #     expected_bin_content.append(Gaussian((bins[i+1] + bins[i])/2, *parameters))
    #
    # #fit_chi_square, p_value_chisrq = chisquare(content, f_exp = expected_bin_content, ddof= int(len(content) - len(parameters))) #Chi_Square(content, expected_bin_content, sigma_data)/()
    #
    # ax_est.plot(x_gauss_fit, y_gauss_fit, color='purple', linewidth=2) #, label=r'$\chi^2/$ndf = %.2f' % fit_chi_square)

    #ax_est.plot([],linewidth=0, label=r'$\hat{I} = \displaystyle\int_{%i}^{%i} \displaystyle\frac{\textrm{d} N}{\textrm{d} \tau} \textrm{d}\tau$' % (lower_lim,upper_lim))
    ax_est.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (p_value))

    ax_est.set_title(r'$\hat{N}(%.0f < \tau < %.0f$ days) distribution' % (lower_lim, upper_lim), fontsize=24)
    ax_est.set_xlabel(r'$\hat{N}(%.0f < \tau < %.0f$ days)' % (lower_lim, upper_lim), fontsize=20)
    ax_est.set_ylabel(r'Arb. units', fontsize=20)
    ax_est.tick_params(axis='both', which='major', labelsize=20)
    ax_est.legend(loc='upper right', fontsize=18)

    fig_est.savefig('./AugerOpenData/Estimator_distribution_histogram_AugerOpenData.pdf')

    print('A 5 sigma excess in our estimator corresponds to', 5*math.sqrt(np.var(estimator_list)))
