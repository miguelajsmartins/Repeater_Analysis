import math
import numpy as np

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

#for data manipulation
import pandas as pd

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)

date = '2015-01-01T00:00:00'
period = '3600.0'
N_total_events = '10000'
N_repeater_events = '65'

tau_rep = pd.read_parquet('./output/Rep_events_with_tau_Date_' + date + '_Period_' + period + '_TotalEvents_' + N_total_events + '_AcceptedRepEvents_' + N_repeater_events + '.parquet', engine='fastparquet')
selection_info = pd.read_parquet('./output/Selection_event_info_Date_' + date + '_Period_' + period + '_TotalEvents_' + N_total_events + '_AcceptedRepEvents_' + N_repeater_events + '.parquet', engine='fastparquet')

print('selection info dataframe types', selection_info.dtypes)

N_events = selection_info.iloc[0,0]

tau_rep = np.log10(np.multiply(tau_rep["tau (s)"].to_numpy(),1/86164))
#tau_ud = np.log10(np.multiply(tau_ud["tau (s)"].to_numpy(),1/86164))

fig_tau = plt.figure(figsize=(10,8)) #create figure
ax_tau = fig_tau.add_subplot(111) #create subplot with a set of axis with

ax_tau.hist(tau_rep,bins=200, density=True, range=[-4,4], alpha = 0.5, label=r'Uniform data w/ rep ($N =$ ' + N_events + ')')
#ax_tau.hist(tau_ud,bins=200, range=[-1.5,5], density=True, alpha = 0.5, label=r'Uniform dist ($N =$ ' + str(N_events) + ')')

print(selection_info.iloc[0,3])

ax_tau.set_title(r'$\log_{10}(\tau)$ distribution for angular window $\Delta \theta =$ ' + str(round(math.degrees(float(selection_info.iloc[0,3])),2)) + '$^{\circ}$',fontsize=24)
ax_tau.set_xlabel(r'$\log_{10}(\tau/ \textrm{sidereal days})$', fontsize=20)
ax_tau.set_ylabel(r'Prob. density', fontsize=20)
ax_tau.tick_params(axis='both', which='major', labelsize=20)
ax_tau.set_xlim([-4,4])
ax_tau.set_yscale('log')
ax_tau.legend(loc='best', fontsize=20)

fig_tau.savefig('./output/Tau_distribution_MockData_86164.pdf')
