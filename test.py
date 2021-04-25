import numpy as np

a=np.empty((3,1))
b=np.ones_like(a)
c=np.hstack((a,b))
c=np.delete(c,0,1)
print(c)

