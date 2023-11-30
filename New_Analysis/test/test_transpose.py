import numpy as np
import numpy.ma as ma

direction = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
indices = np.array([1, 2, 1, 1, 3])

sorted_indices = indices.argsort()

indices = indices[sorted_indices]
direction = direction[sorted_indices]

unique_indices, counts = np.unique(indices, return_counts = True)
slice_positions = np.cumsum(counts)[:-1]

splitted_direction = np.split(direction, slice_positions)

diff_array = np.diff(direction)
filter = diff_array < 2

event_pairs = np.array(list(zip(direction[:-1][filter], direction[1:][filter], diff_array[filter])))

filter = event_pairs[:,-1] < 2

print(filter.shape)
print(event_pairs.shape)
print(direction)
print(diff_array)
print(event_pairs)
