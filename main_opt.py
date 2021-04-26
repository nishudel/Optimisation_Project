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
    #print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #sim.data.ctrl[2]=0.5*np.sin(sim.data.time)
    sim.step()
    '''
    jacp=np.empty((3*model.nv))
    temp=np.empty((3*model.nv))
    jacr=np.empty((1))
    pos=np.empty((1,3))
    pos[0,:]=np.array([0,1,2])
    body_ipos=np.block([[pos],[pos],[pos],[pos]])
    mi=model.body_mass
    bodylist=np.arange(1,4)
    g=9.8
        
    for i in bodylist:        
        mp.functions.mj_jac(model,sim.data,temp,jacr,body_ipos[i,:],i)
    '''
    A,B=f.get_JfootT(model,sim)

    print(A)
    print(B)
    
    #np.delete(H,np.arange(0,6),0)
    #view.render()
    #print(sim.data.time)
    #print(ft_jac_l)
    #print(sim.data.qpos)





