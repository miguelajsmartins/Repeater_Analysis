import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time

inclined_data = pd.read_parquet('AugerOpenData_InclinedEvents_eFit.parquet', engine = 'fastparquet')

timestamp = inclined_data['sd_gpstime'].to_numpy()
energy = inclined_data['sd_energy'].to_numpy()
energy_error = inclined_data['sd_denergy'].to_numpy()
n19 = inclined_data['sd_n19'].to_numpy()
n19_error = inclined_data['sd_dn19'].to_numpy()
theta = inclined_data['sd_theta'].to_numpy()

print(min(theta),max(theta))

print('First event',Time(min(timestamp), format='gps').fits,'Last Event', Time(max(timestamp), format='gps').fits)

#plt.errorbar(energy, n19, energy_error, n19_error, 'o')
#plt.xscale('log')
#plt.yscale('log')

#plt.show()
