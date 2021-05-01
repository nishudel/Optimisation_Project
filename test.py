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
#from centroidal_dynamics import *
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

## Torque  limits
gear=np.zeros((model.nu,2))
gear[:,0]=model.actuator_gear[:,0]
gear[:,1]=gear[:,0]
ctrl_range=np.zeros((model.nu,2))
ctrl_range=model.actuator_ctrlrange
trq_range=np.multiply(gear,ctrl_range)

body_ipos=model.body_ipos
mi=model.body_mass
bodylist=np.arange(1,37)

'''
while True:
    sim.step()
    #if sim.data.time<=5:
    sim=sp.hold_in_air(model,sim)      
    # Target Height of COM
    p=sp.get_1XGT(model,sim)

    view.render()
    print(p)

'''