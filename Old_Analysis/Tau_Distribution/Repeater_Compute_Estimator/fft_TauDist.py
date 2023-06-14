import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

#compute the frequency spectrum of tau distribution
def GetFreqSpectrum(tau_list):

    Ndoublets = len(tau_list)
    amplitude = fft(np.divide(tau_list,86164))
    freq = fftfreq(Ndoublets,10000)[:Ndoublets//2]

    return freq, 2.0/Ndoublets * np.abs(amplitude[0:Ndoublets//2])

tau_filename = "/home/miguelm/Documents/Anisotropies/Repeater_Analysis/DataSets/Vertical/UD_large_stats/Ud_events_with_tau_N_100000_3824455_1.parquet"
df = pd.read_parquet(tau_filename, engine='fastparquet')

tau_list = df['tau (s)'].to_numpy()

frequency, amplitude = GetFreqSpectrum(tau_list)

plt.plot(frequency, amplitude)
plt.yscale('log')
plt.show()
