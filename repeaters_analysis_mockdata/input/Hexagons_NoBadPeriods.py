import pandas as pd
import numpy as np

df = pd.read_parquet('Hexagons.parquet', engine='fastparquet')

gps_time = df.index
n6T5 = df['n6T5'].to_numpy()
n5T5 = df['n5T5'].to_numpy()
bad_period = df['BadPeriodFlag'].to_numpy()

new_n6T5 = np.where(bad_period == 0,0,n6T5)
new_n5T5 = np.where(bad_period == 0,0,n5T5)

list = []

for i in range(len(new_n6T5)):
        list.append([gps_time[i],new_n5T5[i],new_n6T5[i]])

new_df = pd.DataFrame(list, columns=['gps_time','n5T5','n6T5'])

new_df.to_parquet('Hexagons_NoBadStations.parquet', index=False)

print(new_df)
