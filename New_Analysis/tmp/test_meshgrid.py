import numpy as np


x = [1, 2]
y = [3, 4]

matrix = np.transpose(np.array(np.meshgrid(x, y))).reshape(-1, 2)

print(matrix)


rand_number = np.random.randint([0, 0], high = [10, 20], size = (3, 1))

print(rand_number)
