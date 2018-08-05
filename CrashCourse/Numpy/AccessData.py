import numpy as np
my_list = [[1, 2, 3], [4, 5, 6]]
my_array = np.array(my_list)
print(my_array)
print(len(my_array))
print(my_array.shape)
print(my_array[0], my_array[-1], my_array[0, 2], my_array[:, -2])