import mujoco_py as mp
#from mujoco_py import functions 
import numpy as np
import centroidal_dynamics as cdyn
import mosek

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
    #print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #sim.data.ctrl[2]=0.5*np.sin(sim.data.time)
    sim.step()
    #print(sim.data.qM)
    #mp.functions.mj_fullM(model,dst,sim.data.qM)
    #dst=dst.reshape(48,48)
    jacp=cdyn.get_B(model,sim)
    view.render()
    print('hello')
    
    #print(sim.data.qpos)





