import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from random import seed

#fix seed
seed(0)

#define time windows
bg_Tmin = 0
bg_Tmax = 100
signal_Tmin = 10
signal_Tmax = 20
time_window = 5
n_events = 1000

nMax_bg_list = []
nMax_signal_list = []

for i in range(1000):

    background = np.random.random(n_events)*(bg_Tmax - bg_Tmin)
    signal = signal_Tmin + np.random.random(100)*(signal_Tmax - signal_Tmin)
    signal = signal + np.random.choice()

    background = np.sort(background)
    signal = np.sort(signal)

    #print(signal)

    nMax_bg = max([len(background[np.where( ((background - bg_event) < time_window) & ((background - bg_event) > 0) )[0]]) for bg_event in background])
    nMax_signal = max([len(signal[np.where( ((signal - event) < time_window) & ((signal - event) > 0) )[0]]) for event in signal])

    nMax_bg_list.append(nMax_bg)
    nMax_signal_list.append(nMax_signal)

plt.hist(nMax_bg_list, bins = 20)
plt.hist(nMax_signal_list, bins = 20)

#n_max = max(n_diff)

# rate = n_events / (bg_Tmax - bg_Tmin)
# poisson_mu = rate*time_window
#
# print(poisson_mu)
# x = np.arange(0, 2*poisson_mu, 1)
# #n_max_list.append(n_max)
# #time_diff_signal = np.diff(signal)
#
# #print(n_max_list)
#
# bin_contents, bin_edges, _ = plt.hist(n_diff, bins=100, density=True) # range = [10, 25])
# plt.plot(x, poisson.pmf(x,poisson_mu), ms = 2, color = 'tab:orange')
#
# bin_centers = np.array([(bin_edges[i] - bin_edges[i-1]) for i in range(1, len(bin_edges))])
#
# chi2 = sum((bin_contents - poisson.pmf(bin_centers, poisson_mu))**2 / bin_contents ) / len(bin_centers)
#
# print(chi2)

#plt.hist(time_diff_bg, bins = 20, range=[0, 5], color = 'tab:blue', histtype='step')
#plt.hist(time_diff_signal, bins = 20, range=[0, 5], color = 'tab:orange', histtype='step')

plt.show()
