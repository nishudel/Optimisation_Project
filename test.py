import numpy as np


a=2*np.eye(3)
b=5*np.ones((3,1))
c=a@b
print(c[1][0])