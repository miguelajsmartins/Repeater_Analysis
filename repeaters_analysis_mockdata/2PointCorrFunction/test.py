import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

array1 = np.arange(0,100,1)

array2 = np.arange(0,100,1)

sum = 0

for elem in array1:
    for j in range(elem):
        sum+=math.log(elem - j)

new_sum = 0

for elem in array1:
        new_sum+=math.log(math.factorial(elem))

print(sum, new_sum)

repeater_file = '/home/miguelm/Documents/Anisotropies/repeaters_analysis_mockdata/output/TimeOrdered_Events_ExponentialRepeater_Date_2015-01-01T00:00:00_Period_3600.0_TotalEvents_100000_AcceptedRepEvents_67.parquet'

print(pd.read_parquet(repeater_file,engine='fastparquet').head())

#read txt file and plot histogram

#test of arccos
print(math.acos(1), math.acos(-1))

for i in range(2):

    corrfunc_filename = '2PointCorrelationFunction_N_100000_3833100_' + str(i+1) + '.parquet'

    data_2PointCorrFunc = pd.read_parquet(corrfunc_filename, engine='fastparquet')

    print(data_2PointCorrFunc)

    ang_sep_bin_edges = data_2PointCorrFunc['2pointcorr_bin_edges'].to_numpy()
    ang_sep_bin_contents = data_2PointCorrFunc['2pointcorr_bin_content'].to_numpy()

    plt.plot(ang_sep_bin_edges, ang_sep_bin_contents)

plt.show()
