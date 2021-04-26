import mujoco_py as mp
#from mujoco_py import functions 
#from numpy cimport ndarray, float64_t
import numpy as np
from centroidal_dynamics import *
import mosek
import functions as f

mj_path, _ = mp.utils.discover_mujoco()
xml_path ="/home/nishanth/Documents/digit_py/digit-v3/digit-v3.xml" 
model = mp.load_model_from_path(xml_path)
sim = mp.MjSim(model,data=None,nsubsteps=10,udd_callback=None)

#state=mp.MjSimState()
#sim = mp.MjSim(model)
view=mp.MjViewer(sim)
force=np.array([6,1])

# mjtNum* dst
dst=np.empty(2304)
H=np.ones(48)
while True:
    
    
    sim.step()
            




