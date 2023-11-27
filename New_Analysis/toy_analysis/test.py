import numpy as np
import numpy.ma as ma

x = np.array([[1, 1, 2, 1, 4, 5], [1, 3, 4, 3, 4, 1]], dtype = 'float')
y = np.array([[1, 5, 9, 2, 1, 4], [20, 14, 2, 4, 5, 2]], dtype = 'float')

sorted_indices = x.argsort()

sorted_y = ma.masked_array(y, mask = sorted_indices).filled(fill_value = np.nan)

print(y)
print(sorted_y)
