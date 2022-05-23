import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time

auger_data = pd.read_parquet('AugerOpenData_AllEventsWithCuts.parquet', engine = 'fastparquet')

timestamp = auger_data['gpstime'].to_numpy()
energy = auger_data['sd_energy'].to_numpy()
#energy_error = auger_data['sd_denergy'].to_numpy()
#n19 = auger_data['sd_n19'].to_numpy()
#n19_error = auger_data['sd_dn19'].to_numpy()
theta = auger_data['sd_theta'].to_numpy()

print('Total number of events =', len(auger_data.index))
print(min(theta),max(theta))
print('First event',Time(min(timestamp), format='gps').fits,'Last Event', Time(max(timestamp), format='gps').fits)

#plt.errorbar(energy, n19, energy_error, n19_error, 'o')
#plt.xscale('log')
#plt.yscale('log')

#plt.show()
