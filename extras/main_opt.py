import mujoco_py as mp
import numpy as np
from centroidal_dynamics import *
import class_def as cldef
import opt_cvxpy as optcvx

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
    dyn=cldef.centrl_dyn()
    dyn=get_dynamics(model,sim)
    
    dyn=optcvx.get_torque_cvxpy(dyn)
    print(dyn.torque)


    #view.render()
         
'''

sim.step()
torque_t=get_torques(model,sim)

#a=f.run_mosek()

#for i in range(0,20):
#    sim.data.ctrl[i]=torque_t[i]

#print(a)
      
sim.step()
         
'''




