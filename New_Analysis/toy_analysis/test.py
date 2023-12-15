import numpy as np
import numpy.ma as ma

import numpy as np

# Assuming x and y are your arrays with shapes (10, 10, 50)
x = np.array([[1, 2, 3, 7, 4], [1, 2, 3, 4, 9]])
y = np.array([[3, 4, 3, 4, 5], [2, 3, 4, 5, 7]])

count = [np.sum(y[i, :] < x[i, :, np.newaxis], axis = 1) for i in range(x.shape[0])]

#count = [np.sum(y[i] < x[i, :, np.newaxis], axis = 1) for i in range(x.shape[0])]

print(count)
