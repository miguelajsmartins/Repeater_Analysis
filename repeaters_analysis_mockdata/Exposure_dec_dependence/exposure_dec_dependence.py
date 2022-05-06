import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy import integrate

def teo_exposure(theta_max, delta_array, lat_pao):

    exposure = []

    for delta in delta_array:
        x = (math.cos(theta_max) - math.sin(delta)*math.sin(lat_pao))/(math.cos(delta)*math.cos(lat_pao))

        if x > 1:
            hmax = 0
        elif x < -1:
            hmax = math.pi
        else:
            hmax = math.acos(x)

        exposure.append(math.cos(lat_pao)*math.cos(delta)*math.sin(hmax) + hmax*math.sin(lat_pao)*math.sin(delta))

    return exposure


data_ud = pd.read_parquet('/home/miguel/Documents/Repeaters/repeaters_analysis_mockdata/UD_files/TimeOrdered_Accepted_Events_Uniform_Dist_N_100000_3824455_999.parquet', engine='fastparquet')

ud_dec = data_ud['ud_dec'].to_numpy()
ud_sin_dec = np.sin(ud_dec)

#to plot theoretical exposure as a function of dec
lat_pao = math.radians(-35.23)
theta_max = math.radians(60)
theo_dec = np.arange(-math.pi/2,math.pi/2,0.001)
theo_exposure = teo_exposure(theta_max, theo_dec, lat_pao)
theo_exposure_integral = integrate.simpson(theo_exposure,np.sin(theo_dec))

print(theo_exposure_integral)

plt.hist(ud_sin_dec, bins=500, range=[min(ud_sin_dec),max(ud_sin_dec)], density = True)
plt.plot(np.sin(theo_dec), np.divide(theo_exposure,theo_exposure_integral))

plt.show()
