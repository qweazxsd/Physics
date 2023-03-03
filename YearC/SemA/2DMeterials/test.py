import numpy as np 

x = np.arange(1, 7, 2)
len = x.size 


A = np.empty((2,2,len))
A[0, 0, :] = x
A[0, 1, :] = x
A[1, 0, :] = x
A[1, 1, :] = x

y = np.arange(2, 8, 2)
len = y.size


B = np.empty((2,2,len))
B[0, 0, :] = y
B[0, 1, :] = y
B[1, 0, :] = y
B[1, 1, :] = y

C = np.empty(A.shape)

C[0, 0, :] = A[0,0,:]*B[0,0,:] + A[0,1,:]*B[1,0,:]
print(C)
