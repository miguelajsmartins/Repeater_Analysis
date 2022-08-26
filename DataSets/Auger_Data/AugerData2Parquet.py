import numpy as np
import pandas as pd

from astropy.time import Time

filename = 'Arrival_directions_8EeV_Science_2017.dat'
output_filename = 'AugerData_ArrivalDirections_8EeV_Science_2017.parquet'

ra = np.loadtxt(filename, comments='#', usecols=3)
dec = np.loadtxt(filename, comments='#', usecols=2)
time = np.loadtxt(filename, comments='#', usecols=6)

#fake values of theta and energy for the purposes of the analysis
energy = []
theta = []
gpstime = []

for i in range(len(ra)):
    energy.append(10)
    theta.append(70)
    gpstime.append(Time(time[i], format='unix', scale='utc').gps)

df = pd.DataFrame(zip(ra, dec, gpstime, theta, energy), columns=['sd_ra','sd_dec','gpstime', 'sd_theta', 'sd_energy'])

print(df)

df.to_parquet(output_filename,index=False)
