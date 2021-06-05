'''
import class_def as cldef
import cvxpy as cp

def solvecvx(prob,dyn):
    # Change the corresponding matrices
    for i in range(0,dyn.H.x_index.shape[0]):       # Just usign the sparse form
        A_dynamics.value[dyn.H.x_index][dyn.H.y_index]=dyn.H.value[i]
    
    b_t.value[0:6]=dyn.b_t[0:6,0]
'''



import mujoco_py as mp
import numpy as np
from centroidal_dynamics import *
import class_def as cldef
import opt_cvxpy as optcvx
import cvxpy as cp
import support as sp

############### Setting up MuJoCo ###############
mj_path, _ = mp.utils.discover_mujoco()
xml_path ="/home/nishanth/Documents/digit_py/digit-v3/digit-v3.xml" 
model = mp.load_model_from_path(xml_path)
sim = mp.MjSim(model,data=None,nsubsteps=10,udd_callback=None)
view=mp.MjViewer(sim)

p=get_P_cm_i(model,sim)

print(p)
