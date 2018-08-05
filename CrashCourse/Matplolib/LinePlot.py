# basic line plot
import matplotlib.pyplot as plt
import numpy as np

my_array = np.array([1, 2, 3])
plt.plot(my_array)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
