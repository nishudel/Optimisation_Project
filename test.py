import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

a=np.ones((10,1))
numvar=10
xx = [0.] * numvar
a[:,0]=xx
print(a)