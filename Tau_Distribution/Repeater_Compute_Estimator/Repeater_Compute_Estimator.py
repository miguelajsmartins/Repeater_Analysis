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
from scipy.stats import norm

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)
plt.rcParams["figure.autolayout"] = True

#define the gaussian function
def Gaussian(x, mu, sigma, norm, shift):

    return shift + norm*np.exp(-(x - mu) ** 2 / (2* sigma ** 2))

#define the average distribution of tau for many realizations
def AverageTauDist(list_of_histograms):

    list_of_bin_edges = []
    list_of_bin_contents = []

    #print(list_of_histograms)

    for histogram in list_of_histograms:

        #print(histogram)

        bin_contents, bin_edges = histogram

        list_of_bin_edges.append(bin_edges)
        list_of_bin_contents.append(bin_contents)

    #print(len(list_of_bin_edges[0]))

    avg_bin_edges = list_of_bin_edges[0][1:]
    avg_bin_content = []

    nbins = len(avg_bin_edges)

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

            tau_ud = np.divide(df["tau (s)"].to_numpy(), 86164)

            #list_of_log_tau_arrays.append(np.log10(tau_ud))

            list_of_ordered_tau_arrays.append(sorted(tau_ud))

            for tau in tau_ud:
                tau_ud_all.append(tau)

            file_counter+=1

            print(file_counter,'files read!')

    return tau_ud_all, list_of_ordered_tau_arrays

#to get list of ordered taus and list with all taus from files
def FromFiles_to_TauHistograms(path_to_dir, name_of_files, nbins, hist_min, hist_max, lower_lim, upper_lim):

    #lists to hold the histograms with tau distributions
    list_of_tau_histograms = []
    list_of_logtau_histograms = []

    #list to hold the estimators
    N_doublets_below_list = []
    tau_min_list = []

    #to count the number of files read
    file_counter = 0

    #for subdir, dirs, files in os.walk(path_of_dir):
    for filename in os.listdir(path_to_dir):

        f = os.path.join(path_to_dir,filename)

        if os.path.isfile(f) and name_of_files in f:

            df = pd.read_parquet(f, engine='fastparquet')

            tau_array = np.divide(df["tau (s)"].to_numpy(), 86164)

            #lists to hold the histograms
            list_of_tau_histograms.append(np.histogram(tau_array, nbins, range=[math.pow(10,hist_min), math.pow(10,hist_max)]))
            list_of_logtau_histograms.append(np.histogram(np.log10(tau_array), nbins, range=[hist_min, hist_max]))

            #lists with the estimators
            N_doublets_below_list.append(len([tau for tau in tau_array if tau > lower_lim and tau < upper_lim]))
            tau_min_list.append(math.log10(min(tau_array)))

            file_counter+=1

            print(file_counter,'files read!')

    return list_of_tau_histograms, list_of_logtau_histograms, N_doublets_below_list, tau_min_list

#computes the distribution of the estimator
def EstimatorDist(list_of_histograms, lower_lim, upper_lim):

    #estimator distribution for the many isotropic realizations
    estimator_list = []

    for histogram in list_of_histograms:

        bin_content, bin_edges = histogram

        iter = 0
        estimator = 0

        while bin_edges[iter] > lower_lim and bin_edges[iter] < upper_lim:
            estimator+=bin_content[iter]
            iter+=1

        estimator_list.append(estimator)

    return estimator_list

#computes the distribution of the minimum tau
def PValueTauMinDist(list_of_ordered_taus, tau_auger):

    tau_min_list = []

    for tau_list in list_of_ordered_taus:
        tau_min_list.append(tau_list[0])

    #computes the min tau of auger data
    tau_auger_min = min(tau_auger)

    #computes the p-value
    integral_below = len([tau_min for tau_min in tau_min_list if tau_min < tau_auger_min])
    p_value = integral_below/len(tau_min_list)

    if( p_value > 0.5):
        p_value = 1 - p_value

    return tau_min_list, tau_auger_min, p_value

#compute chi^2 for a fit
def Chi_Square(y_data, y_true, sigma):

    r = y_true - y_data

    return sum(( r / sigma)*( r / sigma))

#compute the comulative distribution function
def ComulativeDistFunc(data, nbins):

    pdf_data, pdf_bin_edges = np.histogram(data, nbins)

    cdf = np.cumsum(pdf_data)/sum(pdf_data)

    return pdf_bin_edges[1:], cdf

#compute the comulative distribution function
def ComulativeDistHist(pdf_bin_edges, pdf_data):

    cdf = np.cumsum(pdf_data)/sum(pdf_data)

    return pdf_bin_edges, cdf

#draws an exponential envelop
def LogExpEnvelop(x, rate, x_max):

    return math.log(10)*np.power(10,x)*(rate/ (1 - np.exp(-rate*x_max)) )*np.exp(-rate*np.power(10,x))

#defines the incompatibility between distributions
def Incompatibility(estimator_list_ud, estimator_list_rep, percentile):

    #computes the mean and std of the distribution of UD estimator
    mean_ud = np.mean(estimator_list_ud)
    sigma_ud = math.sqrt(np.var(estimator_list_ud))

    quantile = np.quantile(estimator_list_rep, percentile)

    print('Mean of doublets below from UD distribution', mean_ud)
    print('RMS of doublets below from UD distribution', sigma_ud)
    print('The q(',percentile,') for the distribution of doublets below for Rep is', quantile)

    return (quantile - mean_ud)/sigma_ud

#defines the incompatibility between distributions
def IncompatibilityFromGaussianFit(mean_ud, sigma_ud, mean_rep, sigma_rep, percentile):

    quantile = sigma_rep*norm.ppf(percentile) + mean_rep

    print('FROM FIT: The q(',percentile,') for the distribution of doublets below for Rep is', quantile)

    return (quantile - mean_ud)/sigma_ud

#fit estimator distribution
def FitEstimatorDist(bin_content, bin_edges, estimator_list):

    bin_x = []
    #centers the bin content
    for i in range(len(bin_edges)-1):
        bin_x.append((bin_edges[i+1] + bin_edges[i])/2)

    sigma_data = np.sqrt(bin_content)

    init_parameters = [np.mean(estimator_list), math.sqrt(np.var(estimator_list)), len(estimator_list), 0]

    parameters, covariance = curve_fit(Gaussian, np.array(bin_x), np.array(bin_content), p0=init_parameters) #, sigma = sigma_data, absolute_sigma = True)

    x_gauss_fit = np.arange(min(estimator_list), max(estimator_list), 0.01)
    y_gauss_fit = Gaussian(x_gauss_fit, *parameters)

    parameters_error = np.sqrt(np.diag(covariance))

    print('Mean = ', parameters[0], '+/-', parameters_error[0])
    print('Sigma = ', parameters[1], '+/-', parameters_error[1])
    print('Normalization = ', parameters[2], '+/-', parameters_error[2])
    print('Vertical Shift = ', parameters[3], '+/-', parameters_error[3])

    return x_gauss_fit, y_gauss_fit, parameters, parameters_error, covariance


#set path to dir with uniform dist files
path_to_dir_ud = '../../DataSets/Vertical/UD_AugerOpenData_stats'
path_to_dir_rep = '../../DataSets/Vertical/MockData_Repeaters/Repeater_FixedPosAndDate_AugerOpenData_stats'

#list to hold all tau values from all data sets of isotropy. Note that the limits must be given in sidereal days!!
lower_lim = 0
upper_lim = 2

#read file with tau values for repeater data
PERIOD_OF_REP = '86164'
REP_DATE = 'Date_2015-01-01T00:00:00_Pos_30_20'
N_ACCEPTED_REP_EVENTS = '12'
N_INTENSITY = '12'
N_EXPLOSIONS = float(N_ACCEPTED_REP_EVENTS)/float(N_INTENSITY)

#files with tau distributions
tau_files_ud = 'Ud_events_with_tau'
tau_files_rep = 'REP_VerticalEvents_with_tau_%s_Period_%s_TotalEvents_13068_AcceptedRepEvents_%s_RepIntensity_%s' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY)
list_of_tau_hist_ud, list_of_logtau_hist_ud, N_doublets_below_list_ud, tau_min_list_ud = FromFiles_to_TauHistograms(path_to_dir_ud, tau_files_ud, 200, -3, 4, lower_lim, upper_lim)
list_of_tau_hist_rep, list_of_logtau_hist_rep, N_doublets_below_list_rep, tau_min_list_rep = FromFiles_to_TauHistograms(path_to_dir_rep, tau_files_rep, 200, -3, 4, lower_lim, upper_lim)

#tau_ud_all, list_of_ordered_taus_ud = FromFiles_to_TauDist(path_to_dir_ud, 'Ud_events_with_tau')
#tau_rep_all, list_of_ordered_taus_rep = FromFiles_to_TauDist(path_to_dir_rep, 'Rep_events_with_tau')

#list_of_log_tau_arrays_ud = [np.log10(tau) for tau in list_of_ordered_taus_ud]
#list_of_log_tau_arrays_rep = [np.log10(tau) for tau in list_of_ordered_taus_rep]


#
# df_repeater = pd.read_parquet(path_to_repeaters + 'Rep_events_with_tau_Period_' + PERIOD_OF_REP + '_TotalEvents_100000_AcceptedRepEvents_' + N_ACCEPTED_REP_EVENTS + '_MaxRepIntensity_' + N_INTENSITY + '.parquet', engine='fastparquet')
#
# #store tau values in sidereal days
# tau_repeater = np.divide(df_repeater["tau (s)"].to_numpy(),86164)

#----------------------
# store the tau distribution from auger data and saves important info from the selection info file
#----------------------
# path_to_auger_data = '../results/'
# auger_data_file = 'AugerOpenData_VerticalEvents_with_tau.parquet'
# auger_selection_file = 'AugerOpenData_VerticalEvents_SelectionInfo.parquet'
#
# auger_data = pd.read_parquet(path_to_auger_data + auger_data_file, engine='fastparquet')
# auger_selection_info = pd.read_parquet(path_to_auger_data + auger_selection_file, engine='fastparquet')
#
# #store tau values in sidereal days
# tau_auger = np.divide(auger_data["tau (s)"].to_numpy(), 86164)
#
# #stores important info from auger selection info file
# N_events = int(auger_selection_info.iloc[0]['N_events'])
# theta_min = float(auger_selection_info.iloc[0]['Theta_min'])
# theta_max = float(auger_selection_info.iloc[0]['Theta_max'])
# ang_window = math.degrees(float(auger_selection_info.iloc[0]['Ang_window']))
# t_begin = Time(auger_selection_info.iloc[0]['t_begin'], format='fits').gps
# t_end = Time(auger_selection_info.iloc[0]['t_end'], format = 'fits').gps
#
# #computes the average rate of events in events per sidereal day
# lat_auger = -35.28
# teo_average_rate = 86164 * N_events/(t_end - t_begin) * (1 - math.cos(math.radians(ang_window)))/(1 + math.sin(math.radians(theta_max - lat_auger)))
# auger_avg_rate = 1 / np.mean(tau_auger)
# ud_avg_rate = 1 / np.mean(tau_ud_all)
#
# print('Average rate', teo_average_rate)
# print('Rate from mean auger tau', auger_avg_rate)
# print('Rate from mean uniform dist tau', ud_avg_rate)

#---------------------------------------
# To plot histograms of tau distribution
#---------------------------------------

#perform KS test to verify the compatibity between tau dists with and without repeater
#ks_stat_value, ks_p_value = stats.kstest(tau_rep_all, tau_ud_all)

#plot log10 of tau distributions
fig_tau_log = plt.figure(figsize=(10,8)) #create figure
ax_tau_log = fig_tau_log.add_subplot(111) #create subplot with a set of axis with

#ax_tau_log.hist(np.log10(tau_auger), bins=200, range=[-2,4], alpha = 0.5, label=r'Auger Data: $N_{\textrm{evt}} = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))

log_tau_avg_hist_edges_ud, log_tau_avg_hist_content_ud = AverageTauDist(list_of_logtau_hist_ud)
log_tau_avg_hist_edges_rep, log_tau_avg_hist_content_rep = AverageTauDist(list_of_logtau_hist_rep)

ax_tau_log.plot(log_tau_avg_hist_edges_ud, log_tau_avg_hist_content_ud, label=r'Isotropy')
ax_tau_log.plot(log_tau_avg_hist_edges_rep, log_tau_avg_hist_content_rep, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#ax_tau_log.plot(np.arange(-2,5,0.01), 1000*LogExpEnvelop(np.arange(-2,5,0.01), 5*ud_avg_rate, (t_end - t_begin) / 86164 ), color = 'purple', linestyle='--', label=r'Exponential Envelop',)

#ax_tau_log.plot([],[],lw=0,label=r'KS test: $p$-value = {%.10f}' % ks_p_value)

#---------- CHANGE THIIIIIISSSSSSSSS ---------------------
ang_window = 1 #<--------------
ax_tau_log.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{ 1 sidereal day})$', fontsize = 20)
ax_tau_log.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_tau_log.set_yscale('log')
ax_tau_log.set_ylim(1e-2, 1e3)

ax_tau_log.legend(loc='best', fontsize=18)

fig_tau_log.savefig('./results/Average_log10tau_distribution_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

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
# ax_tau.set_title(r'$\tau$ distribution for angular window $\Psi = 1^\circ$',fontsize=24)
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
ax_cdf_tau_log = fig_cdf_tau_log.add_subplot(111) #reate subplot with a set of axis with

cdf_ud_bin_edges, cdf_ud_content = ComulativeDistHist(log_tau_avg_hist_edges_ud, log_tau_avg_hist_content_ud)
cdf_rep_bin_edges, cdf_rep_content = ComulativeDistHist(log_tau_avg_hist_edges_rep, log_tau_avg_hist_content_rep)

ax_cdf_tau_log.plot(cdf_ud_bin_edges, cdf_ud_content, label=r'Isotropy')
ax_cdf_tau_log.plot(cdf_rep_bin_edges, cdf_rep_content, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))


ax_cdf_tau_log.set_title(r'$N(\log_{10}(\tau))$ for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_cdf_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{1 sidereal day})$', fontsize=20)
ax_cdf_tau_log.set_ylabel(r'Arb. Units', fontsize=20)
ax_cdf_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_cdf_tau_log.set_yscale('log')
#ax_cdf_tau_log.set_ylim(1e-2, 1e4)

ax_cdf_tau_log.legend(loc='best', fontsize = 18)

fig_cdf_tau_log.savefig('./results/Average_log10tau_CDF_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#list with the integration limits given in sidereal days!!!!!
#list_of_integration_lims = [0,1]

#list of p_values for each integration range
list_of_p_values = []

#for i in range(1,len(list_of_integration_lims)):

    #lower_lim = list_of_integration_lims[i-1]
    #upper_lim = list_of_integration_lims[i]

    #estimator value for repeater mock data set
    #auger_data_estimator = len( [tau for tau in tau_auger if (tau > lower_lim and tau < upper_lim)])

    #estimator_list = EstimatorDist(list_of_tau_hist_ud, lower_lim, upper_lim)
    #estimator_list_rep = EstimatorDist(list_of_tau_hist_ud, lower_lim, upper_lim)

    #compute the number of RMS between average and mockdata set point
    #print('Estimator for Mock data set is', abs(np.mean(estimator_list) - auger_data_estimator)/math.sqrt(np.var(estimator_list)),'sigma away from estimator dist mean')

    #integral above the repeater datum point
    #above_rep_data = len([est for est in estimator_list if est >= auger_data_estimator])
    #p_value = above_rep_data/len(estimator_list)

    #print('p-value for tau distribution estimator', p_value)

    #list_of_p_values.append(p_value)

    #draw figure with the estimator
fig_est = plt.figure(figsize=(10,8)) #create figure
ax_est = fig_est.add_subplot(111) #create subplot with a set of axis with

content_ud, bins_ud, _ = ax_est.hist(N_doublets_below_list_ud, bins = max(N_doublets_below_list_ud) - min(N_doublets_below_list_ud) , range=[min(N_doublets_below_list_ud), max(N_doublets_below_list_ud)], alpha=0.5, label='Isotropy')
content_rep, bins_rep, _ = ax_est.hist(N_doublets_below_list_rep, bins = max(N_doublets_below_list_rep) - min(N_doublets_below_list_rep), range=[min(N_doublets_below_list_rep), max(N_doublets_below_list_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#Fit the distribution of estimators for the UD and Rep distributions
print('\n ###### FIT PARAMETERS #######\n')
print('UD distribution:')
x_fit_ud, y_fit_ud, parameters_ud, parameter_error_ud, covariance_ud = FitEstimatorDist(content_ud, bins_ud, N_doublets_below_list_ud)

print('\nRepeater distribution:\n')
x_fit_rep, y_fit_rep, parameters_rep, parameter_error_rep, covariance_rep = FitEstimatorDist(content_rep, bins_rep, N_doublets_below_list_rep)

#plot the fits to the distribution
ax_est.plot(x_fit_ud, y_fit_ud, color='tab:blue', linewidth=2)
ax_est.plot(x_fit_rep, y_fit_rep, color='darkorange', linewidth = 2)

    #ax_est.axvline(auger_data_estimator, 0, max(content), linestyle = 'dashed', color = 'darkorange', label=r'Auger data')

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
    #ax_est.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (p_value))

ax_est.set_title(r'$\hat{N}(%.0f < \tau < %.0f$ days) distribution' % (lower_lim, upper_lim), fontsize=24)
ax_est.set_xlabel(r'$\hat{N}(%.0f < \tau < %.0f$ days)' % (lower_lim, upper_lim), fontsize=20)
ax_est.set_ylabel(r'Arb. units', fontsize=20)
ax_est.tick_params(axis='both', which='major', labelsize=20)
ax_est.legend(loc='upper right', fontsize=18)
ax_est.set_ylim(0,250)

fig_est.savefig('./results/Estimator_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#define the percentile
percentile_doublets = 0.05
mean_gauss_ud = parameters_ud[0]
sigma_gauss_ud = parameters_ud[1]

mean_gauss_rep = parameters_rep[0]
sigma_gauss_rep = parameters_rep[1]

print('\n')
print('The deviation from q(', percentile_doublets, ') of the repeater dist. to the mean of the UD dist. is', Incompatibility(N_doublets_below_list_ud, N_doublets_below_list_rep, percentile_doublets), 'sigma')
print('FROM FIT: The deviation from q(', percentile_doublets, ') of the repeater dist. to the mean of the UD dist. is', IncompatibilityFromGaussianFit(mean_gauss_ud, sigma_gauss_ud, mean_gauss_rep, sigma_gauss_rep, percentile_doublets), 'sigma')

#compute the distribution of tau_min
#tau_min_p_value = PValueTauMinDist(list_of_ordered_taus_ud, tau_auger)

#plot the distribution of tau min
fig_TauMin = plt.figure(figsize=(10,8)) #create figure
ax_TauMin = fig_TauMin.add_subplot(111) #create subplot with a set of axis with

content, bins, _ = ax_TauMin.hist(tau_min_list_ud, bins = 50, range=[-5, 1], alpha=0.5, label='Isotropy')
content_rep, bins_rep, _ = ax_TauMin.hist(tau_min_list_rep, bins = 50, range=[-5,1], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#ax_TauMin.axvline(np.log10(tau_min_auger), 0, max(content), linestyle = 'dashed', color = 'darkorange', label=r'Auger data')

#ax_TauMin.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (tau_min_p_value))

ax_TauMin.set_title(r'$\log_{10}(\tau_{\min})$ distribution', fontsize=24)
ax_TauMin.set_xlabel(r'$\log_{10}(\tau_{\min} / 1\textrm{ sideral day}) $', fontsize=20)
ax_TauMin.set_ylabel(r'Arb. units', fontsize=20)
ax_TauMin.set_ylim(0, 120)
ax_TauMin.tick_params(axis='both', which='major', labelsize=20)
ax_TauMin.legend(loc='upper left', fontsize=18)

fig_TauMin.savefig('./results/TauMin_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))
