import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
selection_file = sys.argv[2]

auger_tau = pd.read_parquet(filename,engine='fastparquet')
auger_selection = pd.read_parquet(selection_file, engine='fastparquet')

print(auger_selection)

tau = np.divide(auger_tau['tau (s)'].to_numpy(), 86164)

plt.hist(np.log10(tau), bins=200, range = [-2,3.5])
plt.yscale('log')
plt.show()
