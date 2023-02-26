import numpy as np 

len = 3 
x = np.linspace(0, 100, len)

A = np.empty((len,2,2))
A[:, 0, 0] = x
A[:, 0, 1] = x
A[:, 1, 0] = x
A[:, 1, 1] = x

print(A)
print(np.ones((2,2)))
