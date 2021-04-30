import cvxpy as cp
import numpy as np
import time

#### Number of Variables
numvars=51                              # T-20 ; F_contact=24 ; e =6 ; t=1
X = cp.Variable(numvars)


### Setting up the matrices and constraints
    
## Objective function- C_transpose matrix                           *** Minimize CX
C=np.zeros((1,51))
C[0][50]=1

## Inequlity Constraints

## 1) Max,Min bound on e to be t  i.e  inf-norm(e)<=t               ***A_infnorm X <=b_infnorm

A_infnorm=np.block([    [np.zeros((6,44)),np.eye(6),-1*np.ones((6,1))],
                        [np.zeros((6,44)),-1*np.eye(6),-1*np.ones((6,1))]])

b_infnorm=np.zeros((12))
    

## 2) Friction Cone Constraints :                                   ***A_frX<=b_fr

#    Used a conservative bound - Restrict Cone to a Pyramid  |F_x+F_y|<F_z/(2*myu) 
            
myu=0.7                                         ######Friction Coefficient #######

# 8 for upper and 8 for lower bound type constraint 
block1=np.array([1, 1, -1/(2*myu)])
block2=np.array([-1, -1, -1/(2*myu)])
A_fr=np.zeros((16,51))

for i in range(0,8):
    colm_addr=np.arange(20+3*(i),20+3*(i+1),1)    # Non-empty columns   
    # Fr_upper_bnd
    A_fr[i][colm_addr[0]:colm_addr[2]+1]=block1
    # Fr_lower_bnd
    A_fr[i+8][colm_addr[0]:colm_addr[2]+1]=block2
                
b_fr=np.zeros((16))

## 3) Torque limits                                                  ***A_trqX<=b_trq
A_trq=np.zeros((40,51))
A_trq[0:20,0:20]=np.eye(20)                     # Max limit    A_trq[20:40,0:20]=-1*np.eye(20)                 # Min limit
b_trq=np.zeros((40))
b_trq[0:20]=trq_range[0:20,0]               # Max limit
b_trq[20:40]=trq_range[0:20,1]              # Min limit

## 4) Constraint on Z - the upper bound on infinity norm             ***A_Z X <=b_z

A_z=np.zeros((1,51))
A_z[0][50]=-1

b_z=0
                         
## 5) Constraint on direction of contact force in Z direction        ***A_fz<=b_fz

A_fz=np.zeros((8,51))

for i in range(0,8):
    A_fz[i][20+i*3+2]=-1                         # Selecting F_z

b_fz=np.zeros((8))

## Equality Constraints                                             ***A_dynamicsX==b_t
A_dynamics=cp.Parameter((6,51))
A_dynamics.value=np.zeros((6,51))
A_dynamics.value[0][0]=-1
print(A_dynamics.value)

b_t=cp.Parameter((6))
b_t.value=np.zeros((6))    

#A_dynamics=np.zeros((6,51))

#for i in range(0,dyn.H.x_index.shape[0]):       # Just usign the sparse form
#    A_dynamics[dyn.H.x_index][dyn.H.y_index]=dyn.H.value[i] 