import numpy as np

y = [[1,2,3], [3,2,1], [2,3,1]]

print(np.argmax(y))
print(np.argmax(y, axis=0))
print(np.argmax(y, axis=1))