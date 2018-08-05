# series
import numpy as np
import pandas as pd

my_array = np.array([1, 2, 3])
row_names = ['a', 'b', 'c']
my_series = pd.Series(my_array, index=row_names)
print(my_series)
