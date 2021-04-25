# Setting up MOSEK for getting torque input for balancing

import sys
import mosek
import numpy as np
import mujoco_py as mp
from centroidal_dynamics import *
from time import process_time



numvars=51          # T-20 ; F_contact=24 ; e =6 ; t=1
numcons=1

def get_torques(model,sim):

    # get_G
    # get_Ahin
    
    






    ## Defining Friction Cone: Use conservative bound - Restrict to Pyramid

    # Step 1: Friction Co-efficient
    myu=0.7

    # Step 2: Define the matrices 
    # |x+y|<= Z/(2*myu) and z>0
    
     # Step 2.1 :    
    block1=np.array([1, 1, -1/(2*myu)])
    fr1=np.zeros((8,24))
    for i in range(0,8):
        fr1[i][i*3:3+i*3]=block1
    
    block2=np.array([-1, -1, -1/(2*myu)])
    fr2=np.zeros((8,24))
    for i in range(0,8):
        fr2[i][i*3:3+i*3]=block2 
    
    # 'b' corresposding to above inequalities
    b_fr=np.zeros((48,1))

    # Step 2.2 :
    block3=np.array([0,0,1])
    fr_z=np.zeros((8,24))
    for i in range(0,8):
        fr_z[i][i*3:3+i*3]=block3

    ## Torque  limits

    gear=np.zeros((model.nu,2))
    gear[:,0]=model.actuator_gear[:,0]
    gear[:,1]=gear[:,0]

    ctrl_range=np.zeros((model.nu,2))
    ctrl_range=model.actuator_ctrlrange

    trq_range=np.multiply(gear,ctrl_range)  

    with mosek.Env() as env:
        with env.Task(0,0) as task:

