# dataframe
import numpy as np
import pandas as pd

my_array = np.array([[1, 2, 3], [4, 5, 6]])
row_names = ['a', 'b']
col_names = ['one', 'two', 'three']

my_data_frame = pd.DataFrame(my_array, index=row_names, columns=col_names)
print(my_data_frame)
print(len(my_data_frame))
print(my_data_frame[0])
print(my_data_frame['one'])
print(my_data_frame.two)
