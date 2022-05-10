import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

auger_tau = pd.read_parquet('AugerOpenData_AllEvents_with_tau.parquet',engine='fastparquet')
auger_selection =  pd.read_parquet('AugerOpenData_AllEvents_SelectionInfo.parquet',engine='fastparquet')

tau = np.divide(auger_tau['tau (s)'].to_numpy(), 86164)

plt.hist(np.log10(tau), bins=200, range = [-1,4])
plt.yscale('log')
plt.show()
