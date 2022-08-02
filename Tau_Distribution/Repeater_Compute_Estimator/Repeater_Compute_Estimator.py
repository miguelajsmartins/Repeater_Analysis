import math
import numpy as np

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition

#for data manipulation
import pandas as pd

#for manipulation and conversion between times
from astropy.time import Time

#to read files
import os

#for fft
from scipy.fft import fft, fftfreq

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

    for histogram in list_of_histograms:

        bin_contents, bin_edges = histogram

        list_of_bin_edges.append(bin_edges)
        list_of_bin_contents.append(bin_contents)

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
def FromFiles_to_TauHistograms(path_to_dir, name_of_files, nbins_log, hist_min_log, hist_max_log, nbins, hist_min, hist_max, lower_tau, upper_tau):

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

        if os.path.isfile(f) and name_of_files in f: # and file_counter < 5:

            df = pd.read_parquet(f, engine='fastparquet')

            tau_array = np.divide(df["tau (s)"].to_numpy(), 86164)

            #lists to hold the histograms
            list_of_tau_histograms.append(np.histogram(tau_array, nbins, range=[hist_min, hist_max]))
            list_of_logtau_histograms.append(np.histogram(np.log10(tau_array), nbins_log, range=[hist_min_log, hist_max_log]))

            #lists with the estimators
            N_doublets_below_list.append(len([tau for tau in tau_array if tau > lower_tau and tau < upper_tau]))
            tau_min_list.append(math.log10(min(tau_array)))

            file_counter+=1

            print(file_counter,'files read!')

    return list_of_tau_histograms, list_of_logtau_histograms, N_doublets_below_list, tau_min_list

#computes the distribution of the estimator
def EstimatorDist(list_of_histograms, lower_tau, upper_tau):

    #estimator distribution for the many isotropic realizations
    estimator_list = []

    for histogram in list_of_histograms:

        bin_content, bin_edges = histogram

        iter = 0
        estimator = 0

        while bin_edges[iter] > lower_tau and bin_edges[iter] < upper_tau:
            estimator+=bin_content[iter]
            iter+=1

        estimator_list.append(estimator)

    return estimator_list

#computes the log likelihood function
def logLikelihood(avg_bin_content, avg_bin_edges, obs_bin_content, max_tau):

    if( len(obs_bin_content) != len(avg_bin_content) ):
        raise ValueError('Size of histograms does not match')

    log_obs_factorial = []
    factorial_sum = []
    obs_times_log_avg = []
    new_avg_bin_content = []

    for i in range(len(obs_bin_content)):

        if( avg_bin_edges[i] > max_tau or avg_bin_content[i] == 0):
            continue

        if( obs_bin_content[i] == 0 ):
            log_obs_factorial.append(0)
        else:
            for j in range(1, int(obs_bin_content[i])+1):
                log_obs_factorial.append(math.log(j))

        factorial_sum.append(sum(log_obs_factorial))
        obs_times_log_avg.append(math.log(avg_bin_content[i])*obs_bin_content[i])
        new_avg_bin_content.append(avg_bin_content[i])

    return -np.sum(new_avg_bin_content) + np.sum(obs_times_log_avg) - sum(factorial_sum)

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

#computes the FFT of the tau distribution
def FFTTauDist(tau_list):



#set path to dir with uniform dist files
path_to_dir_ud = '../../DataSets/Vertical/UD_large_stats'
path_to_dir_rep = '../../DataSets/Vertical/MockData_Repeaters/Repeater_FixedPosAndDate_large_stats'

#list to hold all tau values from all data sets of isotropy. Note that the limits must be given in sidereal days!!
lower_tau = 0
upper_tau = 1

#read file with tau values for repeater data
PERIOD_OF_REP = '86164'
REP_DATE = '2015-01-01T00:00:00'
N_EVENTS = '100000'
N_ACCEPTED_REP_EVENTS = '100'
N_INTENSITY = '100'
N_EXPLOSIONS = float(N_ACCEPTED_REP_EVENTS)/float(N_INTENSITY)

#files with tau distributions
tau_files_ud = 'Ud_events_with_tau'
tau_files_rep = 'Rep_events_with_tau_Date_%s_Period_%s_TotalEvents_%s_AcceptedRepEvents_%s' % (REP_DATE, PERIOD_OF_REP, N_EVENTS, N_ACCEPTED_REP_EVENTS) #, N_INTENSITY)
#tau_files_rep = 'REP_VerticalEvents_with_tau_%s_Period_%s_TotalEvents_%s_AcceptedRepEvents_%s_RepIntensity_%s' % (REP_DATE, PERIOD_OF_REP, N_EVENTS, N_ACCEPTED_REP_EVENTS, N_INTENSITY)

list_of_tau_hist_ud, list_of_logtau_hist_ud, N_doublets_below_list_ud, tau_min_list_ud = FromFiles_to_TauHistograms(path_to_dir_ud, tau_files_ud, 200, -3, 4, 300, 0, 5, lower_tau, upper_tau)
list_of_tau_hist_rep, list_of_logtau_hist_rep, N_doublets_below_list_rep, tau_min_list_rep = FromFiles_to_TauHistograms(path_to_dir_rep, tau_files_rep, 200, -3, 4, 300, 0, 5, lower_tau, upper_tau)

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


#---------- CHANGE THIIIIIISSSSSSSSS ---------------------
ang_window = 1 #<--------------
ax_tau_log.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{ 1 sidereal day})$', fontsize = 20)
ax_tau_log.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_tau_log.set_yscale('log')
ax_tau_log.set_ylim(1e-2, 1e5)

ax_tau_log.legend(loc='best', fontsize=18)

fig_tau_log.savefig('./results/Average_log10tau_distribution_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#--------------------------------------
# plot of tau distributions
#--------------------------------------
fig_tau = plt.figure(figsize=(10,8)) #create figure
ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with

tau_avg_hist_edges_ud, tau_avg_hist_content_ud = AverageTauDist(list_of_tau_hist_ud)
tau_avg_hist_edges_rep, tau_avg_hist_content_rep = AverageTauDist(list_of_tau_hist_rep)

ax_tau.plot(tau_avg_hist_edges_ud, tau_avg_hist_content_ud, label=r'Isotropy')
ax_tau.plot(tau_avg_hist_edges_rep, tau_avg_hist_content_rep, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

ax_tau.set_title(r'$\tau$ distribution for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau.set_xlabel(r'$\tau$ (sidereal days)', fontsize = 20)
ax_tau.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau.tick_params(axis='both', which='major', labelsize=20)
ax_tau.set_yscale('log')
#ax_tau.set_ylim(1e-2, 1e5)
ax_tau.legend(loc='best', fontsize=18)

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

#--------------------------------
# draw figure with the estimator
#--------------------------------
percentile_doublets = 0.1
TS_incompatibility = Incompatibility(N_doublets_below_list_ud, N_doublets_below_list_rep, percentile_doublets)

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

ax_est.set_title(r'$\hat{N}(%.0f < \tau < %.0f$ days) distribution' % (lower_tau, upper_tau), fontsize=24)
ax_est.set_xlabel(r'$\hat{N}(%.0f < \tau < %.0f$ days)' % (lower_tau, upper_tau), fontsize=20)
ax_est.set_ylabel(r'Arb. units', fontsize=20)
ax_est.tick_params(axis='both', which='major', labelsize=20)
ax_est.legend(loc='upper right', fontsize=18)
ax_est.set_ylim(0, 50)
ax_est.text(280, 35, TS_incompatibility, ha='center', va='bottom', fontsize=16, linespacing=1.5, wrap=True, bbox=dict(facecolor='grey', alpha=0.2))

fig_est.savefig('./results/Estimator_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#saves fit parameters
mean_gauss_ud = parameters_ud[0]
sigma_gauss_ud = parameters_ud[1]

mean_gauss_rep = parameters_rep[0]
sigma_gauss_rep = parameters_rep[1]

print('\n')
print('FROM FIT: The deviation from q(', percentile_doublets, ') of the repeater dist. to the mean of the UD dist. is', IncompatibilityFromGaussianFit(mean_gauss_ud, sigma_gauss_ud, mean_gauss_rep, sigma_gauss_rep, percentile_doublets), 'sigma')

#compute the distribution of tau_min
#tau_min_p_value = PValueTauMinDist(list_of_ordered_taus_ud, tau_auger)

#plot the distribution of tau min
fig_TauMin = plt.figure(figsize=(10,8)) #create figure
ax_TauMin = fig_TauMin.add_subplot(111) #create subplot with a set of axis with

content, bins, _ = ax_TauMin.hist(tau_min_list_ud, bins = 50, range=[-7, -1], alpha=0.5, label='Isotropy')
content_rep, bins_rep, _ = ax_TauMin.hist(tau_min_list_rep, bins = 50, range=[-7,-1], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

ax_TauMin.set_title(r'$\log_{10}(\tau_{\min})$ distribution', fontsize=24)
ax_TauMin.set_xlabel(r'$\log_{10}(\tau_{\min} / 1\textrm{ sidereal day}) $', fontsize=20)
ax_TauMin.set_ylabel(r'Arb. units', fontsize=20)
ax_TauMin.set_ylim(0, 120)
ax_TauMin.tick_params(axis='both', which='major', labelsize=20)
ax_TauMin.legend(loc='upper left', fontsize=18)

fig_TauMin.savefig('./results/TauMin_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))

#----------------------------------------------------
# plot the distribution of likelihoods
#----------------------------------------------------
log_likelihood_ud = []
log_likelihood_rep = []

#computes the log likelihood distribution for the isotropic samples
for obs_histogram in list_of_logtau_hist_ud:

    obs_bin_content, obs_bin_edges = obs_histogram
    log_like = logLikelihood(log_tau_avg_hist_content_ud, log_tau_avg_hist_edges_ud, obs_bin_content, math.log10(upper_tau))
    log_likelihood_ud.append(log_like)

    print(len(log_likelihood_ud),'samples done with log_like =', log_like)

#computes the log likelihood distribution for the samples with explosions and iso background
for obs_histogram in list_of_logtau_hist_rep:

    obs_bin_content, obs_bin_edges = obs_histogram
    log_like = logLikelihood(log_tau_avg_hist_content_ud, log_tau_avg_hist_edges_ud, obs_bin_content, math.log10(upper_tau))
    log_likelihood_rep.append(log_like)

    print(len(log_likelihood_rep),'samples done with log_like =', log_like)

#incompatibility between Likelihoods
percentile_like = 0.1
TS_incompatibility = Incompatibility(log_likelihood_ud, log_likelihood_rep, percentile_like)

#figure for loh likelihood
fig_like = plt.figure(figsize=(10,8)) #create figure
ax_like = fig_like.add_subplot(111) #create subplot with a set of axis with

ax_like.hist(log_likelihood_ud, bins = 100, range=[min(log_likelihood_ud), max(log_likelihood_ud)], alpha=0.5, label='Isotropy')
ax_like.hist(log_likelihood_rep, bins = 100, range=[min(log_likelihood_rep), max(log_likelihood_rep)], alpha=0.5, label=r'Isotropy + {%i} events from {%.0f} explosions with $1/\lambda = 1$ day' % (int(N_ACCEPTED_REP_EVENTS), N_EXPLOSIONS))

#plot the fits to the distribution
ax_like.set_title(r'$\ln \mathcal{L}$ distribution for $%.0f < \tau < %.0f$ days' % (lower_tau, upper_tau), fontsize=24)
ax_like.set_xlabel(r'$\ln \mathcal{L}$', fontsize=20)
ax_like.set_ylabel(r'Arb. units', fontsize=20)
ax_like.tick_params(axis='both', which='major', labelsize=20)
ax_like.legend(loc='upper right', fontsize=18)
ax_like.set_ylim(0, 50)
ax_like.text(-2500, 35, TS_incompatibility, ha='center', va='bottom', fontsize=16, linespacing=1.5, wrap=True, bbox=dict(facecolor='grey', alpha=0.2))

fig_like.savefig('./results/LogLikelihood_distribution_histogram_%s_RepPeriod_%s_TotalIntensity_%s_RepIntensity_%s.pdf' % (REP_DATE, PERIOD_OF_REP, N_ACCEPTED_REP_EVENTS, N_INTENSITY))
