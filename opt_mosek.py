# Setting up MOSEK for getting torque input for balancing

import sys
import mosek
import numpy as np
import mujoco_py as mp
from centroidal_dynamics import *
from time import process_time





def get_torques(model,sim):
    
    numvars=51          # T-20 ; F_contact=24 ; e =6 ; t=1
    numcons=1
    inf=0.0

    ## Objective function- C matrix
    C=np.zeros(1,51)
    C[0][50]=1


    ### Equality Constraint Matrix
    
    # [ lambda_T lambda_Foot ones zeros]  6x(20 24 6 1)
    lambda_T,lambda_F=get_lambdaTF(model,sim)

    H=np.block([lambda_T, lambda_F, np.eye(6) ,np.zeros((6,1))])

    # 'b' term for the above equality constraint
    b_lambda=get_bt(sim,model)
    
    ## Defining Friction Cone Constraints: Use conservative bound - Restrict to Pyramid

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

    ## Torque  limit Constraints

    gear=np.zeros((model.nu,2))
    gear[:,0]=model.actuator_gear[:,0]
    gear[:,1]=gear[:,0]

    ctrl_range=np.zeros((model.nu,2))
    ctrl_range=model.actuator_ctrlrange

    trq_range=np.multiply(gear,ctrl_range)  

    with mosek.Env() as env:
        with env.Task(0,0) as task:

            # Decision Variables:
            task.appendvars(numvars)

            # Set the C matrix and Variable limits
            for i in range(0,51):
                # For objective
                task.putcj(i,C[i])              

                # Bounds on decision variables
                if i<=19:
                    task.putvarbound(i,mosek.boundkey.ra,trq_range[i][0],trq_range[i][1])     # Torque limits
                elif i==50:                                                 # The bound on error i.e t
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                elif i>=20 and i<=43 and (i+1)%3==0:                        # Z-Ground reaction force
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                else:                                                       # Otherwise
                    task.putvarbound(i,mosek.boundkey.fr,-inf,+inf)


            # Equality Constraints
            numcon=0                        # Keep track of the number of constraints already used
           
            val=np.empty((1,numvars))       # Temperory memory for a row 
            j=np.arange(numvars)            # To indicate all columns 

            numcon_eq=6                     # Number of equality constraints
            

            task.appendcons(numcon_eq)
            fx=mosek.boundkey.fx            # Setting up boundkey for equality constraint i.e HX=B

            for i in range(0+numcon,6+numcon):
                val=H[i,:]                  # Collect a row
                task.putarow(i,j,val)

                task.putconbound(i,fx,b_lambda[i][0],b_lambda[i][0]) # Setting upper=lower bound for equality


            numcon+=numcon_eq    

            # Inequality Constraints - Append to earlier Constraints

            # Max,Min bound on e to be t  i.e  inf-norm(e)<=t

            numcon_infnorm=12       # 6 for lower and 6 for upper bounds
            
            task.appendcons(numcon_infnorm)

            up=mosek.boundkey.up    # Bound key- Upper bound 

            A_infnorm=np.block([    [np.zeros((6,44)),np.eye(6),-1*np.ones((6,1))],
                                    [np.zeros((6,44)),-1*np.eye(6),-1*np.ones((6,1))]])
            
            j=np.arange(44,numvars,1)               # Non-empty colm of A_infnorm

            for i in range(0+numcon,12+numcon):
                val=A_infnorm[i,44:numvars]         # Collect nonempty part of each row
                task.putarow(i,j,val)

                task.putconbound(i,up,-inf,0)       # Setting upper bound as zero

            numcon+=numcon_infnorm

            # Friction Cone Constraints : |F_x+F_y|<F_z/(2*myu)

            numcon_fr=16                            # 8 for upper and 8 for lower bound type constraint
            block1=np.array([1, 1, -1/(2*myu)])
            block2=np.array([-1, -1, -1/(2*myu)])

            for i in range(0+numcon,8+numcon):
                colm_addr=np.arange(20+3*i,20+3*(i+1),1)
                # Fr_upper_bnd
                task.putarow(i,colm_addr,block1)
                task.putconbound(i,up,-inf,0)
                # Fr_lower_bnd
                task.putarow(i+8,colm_addr,block2)
                task.putconbound(i+8,up,-inf,0)




            














