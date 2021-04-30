'''
import class_def as cldef
import cvxpy as cp

def solvecvx(prob,dyn):
    # Change the corresponding matrices
    for i in range(0,dyn.H.x_index.shape[0]):       # Just usign the sparse form
        A_dynamics.value[dyn.H.x_index][dyn.H.y_index]=dyn.H.value[i]
    
    b_t.value[0:6]=dyn.b_t[0:6,0]
'''


'''
import mujoco_py as mp
import numpy as np
from centroidal_dynamics import *
import class_def as cldef
import opt_cvxpy as optcvx
import cvxpy as cp

############### Setting up MuJoCo ###############
mj_path, _ = mp.utils.discover_mujoco()
xml_path ="/home/nishanth/Documents/digit_py/digit-v3/digit-v3.xml" 
model = mp.load_model_from_path(xml_path)
sim = mp.MjSim(model,data=None,nsubsteps=10,udd_callback=None)
view=mp.MjViewer(sim)

## Torque  limits
gear=np.zeros((model.nu,2))
gear[:,0]=model.actuator_gear[:,0]
gear[:,1]=gear[:,0]
ctrl_range=np.zeros((model.nu,2))
ctrl_range=model.actuator_ctrlrange
trq_range=np.multiply(gear,ctrl_range)



print(model.nv)
print(model.nq)
'''

import numpy as np

def get_B():
    B=np.zeros((54,20))
    
    row_addr=[0,1,2,6,10,14]     # Non-zero row address for left,right legs
    
    # Legs
    j=0                         # Actuator number
    for i in row_addr:
        B[6+i][j]=1             # Left Leg  6 => Base
        B[26+i][j+6]=1          # Right Leg 26=> Base + Left Leg 
        j=j+1

    j=12                        # Legs make up 12 actuator

    # Arms
    for i in range(0,8):
        B[46+i][j]=1            # Left arm followed by right arm 46 => Base + Both Legs
        j=j+1

    return B                    # (nvx20)

B=get_B()

np.savetxt("B.csv",B, delimiter=",")