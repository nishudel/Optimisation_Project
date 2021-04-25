import numpy as np


a=2*np.eye(3)
b=5*np.ones((3,1))
b=a[0,0:3]
d=2
print(b)