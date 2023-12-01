import numpy as np
import numpy.ma as ma

x = np.array([[1., 2.], [3., 4.], [5, 6]])
y = np.array([1., 2.])

diff = x - y[np.newaxis,:]

print(diff)

is_null = diff == 0

result = ma.masked_array(x, mask = np.logical_not(is_null)).filled(fill_value = np.nan)

print(result)
