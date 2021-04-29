# Setting up MOSEK for getting torque input for balancing

import sys
import mosek
import numpy as np
import mujoco_py as mp
from centroidal_dynamics import *
from time import process_time

def get_putaijlist(H):                  # Returns the non-empty set of index and vals of nparray H
    index=np.transpose(np.nonzero(H))
    x_index=index[:,0]
    y_index=index[:,1]
    val=H[index[:,0],index[:,1]]

    return x_index,y_index,val


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def get_torques(model,sim):
    torque=np.empty((20,1))     # Store torques as output
    '''
    inf=0.0
    numvars=51          # T-20 ; F_contact=24 ; e =6 ; t=1

    ### Objective function- C matrix
    C=np.zeros((1,51))
    C[0][50]=1

    ### Equality Constraint Matrix
            
    ## Dynamics: [ lambda_T lambda_Foot ones zeros]  6x(20 24 6 1)
    lambda_T,lambda_F=get_lambdaTF(model,sim)

    H=np.block([lambda_T, lambda_F, np.eye(6) ,np.zeros((6,1))])

    # 'b' term for the above equality constraint
    b_lambda=get_bt(model,sim)
            
    ## Torque  limits
    gear=np.zeros((model.nu,2))
    gear[:,0]=model.actuator_gear[:,0]
    gear[:,1]=gear[:,0]

    ctrl_range=np.zeros((model.nu,2))
    ctrl_range=model.actuator_ctrlrange
            
    trq_range=np.multiply(gear,ctrl_range)

    A_infnorm=np.block([    [np.zeros((6,44)),np.eye(6),-1*np.ones((6,1))],
                            [np.zeros((6,44)),-1*np.eye(6),-1*np.ones((6,1))]])
            

    myu=0.7                                         ######Friction Coefficient #######
    block1=np.array([1, 1, -1/(2*myu)])
    block2=np.array([-1, -1, -1/(2*myu)])
    '''
    with mosek.Env() as env:
        with env.Task(0,0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            inf=0.0
            numvars=51          # T-20 ; F_contact=24 ; e =6 ; t=1

            ### Objective function- C matrix
            C=np.zeros((1,51))
            C[0][50]=1

            ### Equality Constraint Matrix
                    
            ## Dynamics: [ lambda_T lambda_Foot ones zeros]  6x(20 24 6 1)
            lambda_T,lambda_F=get_lambdaTF(model,sim)

            H=np.block([lambda_T, lambda_F, np.eye(6) ,np.zeros((6,1))])

            # 'b' term for the above equality constraint
            b_lambda=get_bt(model,sim)
                    
            ## Torque  limits
            gear=np.zeros((model.nu,2))
            gear[:,0]=model.actuator_gear[:,0]
            gear[:,1]=gear[:,0]

            ctrl_range=np.zeros((model.nu,2))
            ctrl_range=model.actuator_ctrlrange
                    
            trq_range=np.multiply(gear,ctrl_range)

            A_infnorm=np.block([    [np.zeros((6,44)),np.eye(6),-1*np.ones((6,1))],
                                    [np.zeros((6,44)),-1*np.eye(6),-1*np.ones((6,1))]])
                    

            myu=0.7                                         ######Friction Coefficient #######
            block1=np.array([1, 1, -1/(2*myu)])
            block2=np.array([-1, -1, -1/(2*myu)])


          
            # Decision Variables:
            
            task.appendvars(numvars)    

            # Set the C matrix and Variable limits
            for i in range(0,51):
                # For objective
                task.putcj(i,C[0][i])              

                # Bounds on decision variables
                if i<=19:
                    task.putvarbound(i,mosek.boundkey.ra,trq_range[i][0],trq_range[i][1])     # Torque limits
                elif i==50:                                                 # The bound on error i.e t
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                elif i>=20 and i<=43 and (i-1)%3==0:                        # Z-Ground reaction force>=0
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                else:                                                       # Otherwise
                    task.putvarbound(i,mosek.boundkey.fr,-inf,+inf)
            
            ## Equality Constraints for Dynamics

            numcon=0                        # Keep track of the number of constraints already used
           
            val=np.empty((1,numvars))       # Temperory memory for a row 
            j=np.arange(numvars)            # To indicate all columns 

            numcon_eq=6                     # Number of equality constraints            

            task.appendcons(numcon_eq)

            fx=mosek.boundkey.fx                                        # Setting up boundkey for equality constraint i.e HX=B

            for i in range(0+numcon,6+numcon):
                val=H[i,:]                                              # Collect full row
                task.putarow(i,j,val)                                   # Set H matrix
                task.putconbound(i,fx,b_lambda[i][0],b_lambda[i][0])    # Setting upper=lower bound for equality
            
            numcon+=numcon_eq    

            ### Inequality Constraints 

            ## 1) Max,Min bound on e to be t  i.e  inf-norm(e)<=t

            numcon_infnorm=12       # 6 for lower and 6 for upper bounds
            
            task.appendcons(numcon_infnorm)

            up=mosek.boundkey.up    # Bound key- Upper bound 
            j=np.arange(44,numvars,1)               # Non-empty colm of A_infnorm
            val=np.empty((1,7))     

            for i in range(0+numcon,12+numcon):
                val=A_infnorm[i-numcon,44:numvars]         # Collect nonempty part of each row
                task.putarow(i,j,val)               
                task.putconbound(i,up,-inf,0)       # Setting upper bound as zero
            
            numcon+=numcon_infnorm

            ## 2) Friction Cone Constraints : 
            #    Used a conservative bound - Restrict Cone to a Pyramid  |F_x+F_y|<F_z/(2*myu) 
            
            

            numcon_fr=16                                    # 8 for upper and 8 for lower bound type constraint 
            task.appendcons(numcon_fr)

            for i in range(0+numcon,8+numcon):
                colm_addr=np.arange(20+3*(i-numcon),20+3*(i-numcon+1),1)    # Non-empty columns   
                # Fr_upper_bnd
                task.putarow(i,colm_addr,block1)
                task.putconbound(i,up,-inf,0)
                # Fr_lower_bnd
                task.putarow(i+8,colm_addr,block2)
                task.putconbound(i+8,up,-inf,0)
            
            numcon+=numcon_fr

            # Define the nature of task
            task.putobjsense(mosek.objsense.minimize)
           
            # Solve the problem
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            
            # Check Status 
            solsta = task.getsolsta(mosek.soltype.bas)

            
            if (solsta == mosek.solsta.optimal):

                xx = [0.] * 20
                task.getxxslice(mosek.soltype.bas,0,20,xx) # storing just the torques
                #torque=xx
            elif (solsta == mosek.solsta.dual_infeas_cer or solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
          
           




    #print(torque.shape)

    

    return xx      













            














