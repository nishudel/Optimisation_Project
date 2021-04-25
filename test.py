import numpy as np


a=2*np.eye(5)
b=[0,1,3]
for i in range(0,4):
    for j in b:
        a[1+i][b]=0
print(a)