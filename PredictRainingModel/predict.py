import numpy as np
import pandas as pd

# read 23 years weather report 
df = pd.read_csv('2000_2023.csv')

# precipitation greater than or equal to 2.5mm means rain
df['rain'] = df['precipitation'].apply(lambda x: 1 if x >= 2.5 else 0)

print(df.head())
