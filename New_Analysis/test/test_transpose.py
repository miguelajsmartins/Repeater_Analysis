import numpy as np
import numpy.ma as ma

array = np.array([1., 2., 3., 4., 5., 6.])
filter = np.array([3., 5.])

tiled_array = np.tile(array, (filter.shape[0], 1))

below_filter = array < filter[:,np.newaxis]

masked_array = ma.masked_array(tiled_array, mask = np.logical_not(below_filter)).filled(fill_value = np.nan)

print(masked_array)
