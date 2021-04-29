import mujoco_py as mp
import numpy as np
from centroidal_dynamics import *
from opt_mosek import *
import mosek
import functions as f
import class_def as cldef

#dyn=cldef.centrl_dyn(np.ones(20),0*np.ones((5,5)))
#print(type(dyn))
#print(dyn.B.x_index)


## nu
nu=model.nu

## Torque  limits
gear=np.zeros((model.nu,2))
gear[:,0]=model.actuator_gear[:,0]
gear[:,1]=gear[:,0]
ctrl_range=np.zeros((model.nu,2))
ctrl_range=model.actuator_ctrlrange
trq_range=np.multiply(gear,ctrl_range)

## Dynamics: [ lambda_T lambda_Foot ones zeros]  6x(20 24 6 1)
lambda_T,lambda_F=get_lambdaTF(model,sim)
H=np.block([lambda_T, lambda_F, np.eye(6) ,np.zeros((6,1))])

## b_t
b_t=get_bt(model,sim)

dyn_data=cldef.centrl_dyn(nu,trq_range,H,b_t)




