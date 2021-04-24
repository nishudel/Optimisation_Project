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
    





    with mosek.Env() as env:
        with env.Task(0,0) as task:

