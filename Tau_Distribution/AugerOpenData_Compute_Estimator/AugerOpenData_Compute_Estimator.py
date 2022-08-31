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

#draws an exponential envelop
def LogExpEnvelop(x, rate, x_max):

    return math.log(10)*np.power(10,x)*(rate/ (1 - np.exp(-rate*x_max)) )*np.exp(-rate*np.power(10,x))

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

#set path to dir with uniform dist files
path_to_dir_ud = '../../DataSets/Vertical/UD_AugerOpenData_stats'

#define tau upper and lower limits with tau in days
lower_tau = 0
upper_tau = 1

#list to hold all tau values from all data sets of isotropy
tau_ud_all, list_of_ordered_taus_ud = FromFiles_to_TauDist(path_to_dir_ud, 'UD_VerticalEvents_with_tau')

#list with values of log10(tau)
list_of_log_tau_arrays = [np.log10(tau) for tau in list_of_ordered_taus_ud]

#----------------------
# store the tau distribution from auger data and saves important info from the selection info file
#----------------------
path_to_auger_data = '../results/'
auger_data_file = 'AugerOpenData_VerticalEvents_with_tau.parquet'
auger_selection_file = 'AugerOpenData_VerticalEvents_SelectionInfo.parquet'

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
lat_auger = -35.28
teo_average_rate = 86164 * N_events/(t_end - t_begin) * (1 - math.cos(math.radians(ang_window)))/(1 + math.sin(math.radians(theta_max - lat_auger)))
auger_avg_rate = 1 / np.mean(tau_auger)
ud_avg_rate = 1 / np.mean(tau_ud_all)

print('Average rate', teo_average_rate)
print('Rate from mean auger tau', auger_avg_rate)
print('Rate from mean uniform dist tau', ud_avg_rate)

#---------------------------------------
# To plot histograms of tau distribution
#---------------------------------------

#perform KS test to verify the compatibity between tau dists with and without repeater
ks_stat_value, ks_p_value = stats.kstest(tau_auger, tau_ud_all)

#plot log10 of tau distributions
fig_tau_log = plt.figure(figsize=(10,8)) #create figure
ax_tau_log = fig_tau_log.add_subplot(111) #create subplot with a set of axis with

logtau_auger_hist_content, logtau_auger_hist_edges, _ = ax_tau_log.hist(np.log10(tau_auger), bins=200, range=[-3,4], alpha = 0.5, label=r'Auger Data: $N_{\textrm{evt}} = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))

log_tau_avg_hist_edges, log_tau_avg_hist_content = AverageTauDist(list_of_log_tau_arrays, 200, -3, 4)

ax_tau_log.plot(log_tau_avg_hist_edges, log_tau_avg_hist_content, label=r'Isotropy')
#ax_tau_log.plot(np.arange(-2,5,0.01), 1000*LogExpEnvelop(np.arange(-2,5,0.01), 5*ud_avg_rate, (t_end - t_begin) / 86164 ), color = 'purple', linestyle='--', label=r'Exponential Envelop',)

ax_tau_log.plot([],[],lw=0,label=r'KS test: $p$-value = {%.2f}' % ks_p_value)

ax_tau_log.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{ 1 sidereal day})$', fontsize=20)
ax_tau_log.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_tau_log.set_yscale('log')
ax_tau_log.set_ylim(1e-3, 1e3)

ax_tau_log.legend(loc='best', fontsize=18)

fig_tau_log.savefig('./AugerVerticalData/Average_log10tau_distribution_AugerVerticalData.pdf')

#--------------------------------------
# plot of tau distributions
#--------------------------------------
fig_tau = plt.figure(figsize=(10,8)) #create figure
ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with

ax_tau.hist(tau_auger, bins=200, range=[0,6000], alpha = 0.5, label=r'Auger Data: $N = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))

tau_avg_hist_edges, tau_avg_hist_content = AverageTauDist(list_of_ordered_taus_ud, 200, 0, 6000)

ax_tau.plot(tau_avg_hist_edges, tau_avg_hist_content, label=r'Isotropy')

ax_tau.set_title(r'$\tau$ distribution for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_tau.set_xlabel(r'$\tau$ (sidereal days)', fontsize=20)
ax_tau.set_ylabel(r'Number of pairs', fontsize=20)
ax_tau.tick_params(axis='both', which='major', labelsize=20)
ax_tau.set_xlim([0,6000])
ax_tau.set_yscale('log')
ax_tau.set_ylim(5e-1,1e3)

ax_tau.legend(loc='upper left', fontsize=18)

#for inset
ax_tau_inset = fig_tau.add_axes([0,0,1,1])
inset_pos = InsetPosition(ax_tau, [0.6,0.5,0.37,0.4])
ax_tau_inset.set_axes_locator(inset_pos)

ax_tau_inset.hist(tau_auger, bins=100, range=[0,10], alpha = 0.5)

tau_avg_hist_edges_inset, tau_avg_hist_content_inset = AverageTauDist(list_of_ordered_taus_ud, 100, 0, 10)

ax_tau_inset.plot(tau_avg_hist_edges_inset, tau_avg_hist_content_inset)

ax_tau_inset.set_xlabel(r'$\tau$ (sidereal days)', fontsize=16)
ax_tau_inset.set_ylabel(r'Number of pairs', fontsize=16)
ax_tau_inset.tick_params(axis='both', which='major', labelsize=16)
#ax_tau_inset.set_xlim([0,10])
#ax_tau.set_yscale('log')
#ax_tau_inset.set_ylim(1e-2,1e2)

fig_tau.savefig('./AugerVerticalData/Average_tau_distribution_AugerVerticalData.pdf')

#---------------------------------------
# plot of cdf of log tau
#---------------------------------------
fig_cdf_tau_log = plt.figure(figsize=(10,8)) #create figure
ax_cdf_tau_log = fig_cdf_tau_log.add_subplot(111) #reate subplot with a set of axis with

cdf_rep_bin_edges, cdf_rep_content = ComulativeDistFunc(np.log10(tau_auger), 100)
cdf_ud_bin_edges, cdf_ud_content = ComulativeDistFunc(np.log10(tau_ud_all), 100)

ax_cdf_tau_log.plot(cdf_rep_bin_edges, cdf_rep_content, label=r'Auger Data: $N_{\textrm{evt}} = {%i}$, ${%.0f}^\circ < \theta < {%.0f}^\circ$' % (N_events, theta_min, theta_max))
ax_cdf_tau_log.plot(cdf_ud_bin_edges, cdf_ud_content, label=r'Isotropy')

ax_cdf_tau_log.set_title(r'$N(\log_{10}(\tau))$ for angular window $\Psi = {%.0f}^\circ$' % ang_window, fontsize=24)
ax_cdf_tau_log.set_xlabel(r'$\log_{10}(\tau/ \textrm{sidereal days})$', fontsize=20)
ax_cdf_tau_log.set_ylabel(r'Arb. Units', fontsize=20)
ax_cdf_tau_log.tick_params(axis='both', which='major', labelsize=20)
ax_cdf_tau_log.set_yscale('log')
#ax_cdf_tau_log.set_ylim(1e-2, 1e4)

ax_cdf_tau_log.legend(loc='best', fontsize=18)

fig_cdf_tau_log.savefig('./AugerVerticalData/Average_log10tau_CDF_AugerVerticalData.pdf')

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

    content, bins, _ = ax_est.hist(estimator_list, bins = max(estimator_list) - min(estimator_list), range=[min(estimator_list), max(estimator_list)], alpha=0.7, color = 'tab:orange', label='Isotropy')
    #content_rep, bins_rep, _ = ax_est.hist(estimator_list_rep, bins = max(estimator_list_rep) - min(estimator_list_rep), range=[min(estimator_list_rep), max(estimator_list_rep)], alpha=0.5, label='Repeater distribution')

    ax_est.axvline(auger_data_estimator, 0, max(content), linestyle = 'dashed', color = 'tab:blue', label=r'Auger data')

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

    ax_est.set_title(r'$N_{%.0f^\circ, %.0f \textrm{ day}}$-distribution' % (ang_window, upper_lim), fontsize = 24)
    ax_est.set_xlabel(r'$N_{%.0f^\circ, %.0f \textrm{ day}}$' % (ang_window, upper_lim), fontsize = 20)
    ax_est.set_ylabel(r'Arb. units', fontsize=20)
    ax_est.tick_params(axis='both', which='major', labelsize=20)
    ax_est.legend(loc='upper right', fontsize=18)

    fig_est.savefig('./AugerVerticalData/Estimator_distribution_histogram_AugerVerticalData.pdf')

    print('A 5 sigma excess in our estimator corresponds to', 5*math.sqrt(np.var(estimator_list)))

#-----------------------------------------
#   compute the distribution of tau_min
#-----------------------------------------
tau_min_ud, tau_min_auger, tau_min_p_value = PValueTauMinDist(list_of_ordered_taus_ud, tau_auger)

#plot the distribution of tau min
fig_TauMin = plt.figure(figsize=(10,8)) #create figure
ax_TauMin = fig_TauMin.add_subplot(111) #create subplot with a set of axis with

content, bins, _ = ax_TauMin.hist(np.log10(tau_min_ud), bins = 50, color = 'tab:orange', alpha = 0.7, range=[min(np.log10(tau_min_ud)), max(np.log10(tau_min_ud))], label='Isotropy')

ax_TauMin.axvline(np.log10(tau_min_auger), 0, max(content), linestyle = 'dashed', color = 'tab:blue', label=r'Auger data')
ax_TauMin.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (tau_min_p_value))

ax_TauMin.set_title(r'$\log_{10}(\tau_{\min})$ distribution', fontsize=24)
ax_TauMin.set_xlabel(r'$\log_{10}(\tau_{\min} / 1\textrm{ sideral day}) $', fontsize=20)
ax_TauMin.set_ylabel(r'Arb. units', fontsize=20)
ax_TauMin.tick_params(axis='both', which='major', labelsize=20)
ax_TauMin.legend(loc='upper left', fontsize=18)

fig_TauMin.savefig('./AugerVerticalData/TauMin_distribution_histogram_AugerVerticalData.pdf')

#--------------------------------------------
# compute the log likelihood distribution
#--------------------------------------------

loglikelihood_ud = []
loglikelihood_auger = logLikelihood(log_tau_avg_hist_content, log_tau_avg_hist_edges, logtau_auger_hist_content, math.log10(upper_tau))

#computes loglikelihood distribution from samples of isotropy
for logtau_list in list_of_log_tau_arrays :

    obs_bin_contents, obs_bin_edges = np.histogram(logtau_list, bins=200, range=[-3,4])

    log_like = logLikelihood(log_tau_avg_hist_content, log_tau_avg_hist_edges, obs_bin_contents, math.log10(upper_tau))
    loglikelihood_ud.append(log_like)

    print(len(loglikelihood_ud),'samples done with log_like =', log_like)

#compute p-value for auger likelihood
if (np.mean(loglikelihood_ud) > loglikelihood_auger):
    loglike_below_data = len([log_like for log_like in loglikelihood_ud if log_like < loglikelihood_auger])
    pvalue_loglike = loglike_below_data/len(loglikelihood_ud)

else:
    loglike_below_data = len([log_like for log_like in loglikelihood_ud if log_like > loglikelihood_auger])
    pvalue_loglike = loglike_below_data/len(loglikelihood_ud)

#plot the likelihood distribution
fig_like = plt.figure(figsize=(10,8)) #create figure
ax_like = fig_like.add_subplot(111) #create subplot with a set of axis with

ax_like.hist(loglikelihood_ud, bins = 100, color='tab:orange', alpha = 0.7, range=[min(loglikelihood_ud), max(loglikelihood_ud)], label='Isotropy')
ax_like.axvline(loglikelihood_auger, 0, 1e3, linestyle = 'dashed', color = 'tab:blue', label=r'Auger data')
ax_like.plot([],linewidth=0, label=r'$p$-value = {%.3f}' % (pvalue_loglike))
#plot the fits to the distribution
ax_like.set_title(r'$\ln \mathcal{L}_{%.0f^\circ, %.0f \textrm{ day}}$-distribution' % (ang_window, upper_tau), fontsize=24)
ax_like.set_xlabel(r'$\ln \mathcal{L}_{%.0f^\circ, %.0f \textrm{ day}}$' % (ang_window, upper_tau), fontsize=20)
ax_like.set_ylabel(r'Arb. units', fontsize=20)
ax_like.tick_params(axis='both', which='major', labelsize=20)
ax_like.legend(loc='upper left', fontsize=18)
ax_like.set_yscale('log')
#ax_like.set_ylim(0, 50)
#ax_like.text(-91000, 35, TS_incompatibility, ha='center', va='bottom', fontsize=16, linespacing=1.5, wrap=True, bbox=dict(facecolor='grey', alpha=0.2))

fig_like.savefig('./AugerVerticalData/LogLikelihood_distribution_histogram_AugerVerticalData.pdf')
