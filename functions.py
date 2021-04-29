import sys
import mosek
import mujoco_py as mp
from centroidal_dynamics import *
import class_def 

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# Returns the non-empty set of index and vals of nparray H
def get_putaijlist(H):                  
    index=np.transpose(np.nonzero(H))
    x_index=index[:,0]
    y_index=index[:,1]
    val=H[index[:,0],index[:,1]]

    return x_index,y_index,val


def get_torques(dyn,torque):
    #torque=np.zeros(20)
    with mosek.Env() as env:
        with env.Task(0,0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Decision Variables:
            numvars=51          # T-20 ; F_contact=24 ; e =6 ; t=1
            task.appendvars(numvars)
            
            inf=0.0

            ### Objective function- C matrix
            C=np.zeros((1,51))
            C[0][50]=1
 
            # Set the C matrix and Variable limits
            for i in range(0,51):
                # For objective
                task.putcj(i,C[0][i])              

                # Bounds on decision variables
                if i<=19:
                    task.putvarbound(i,mosek.boundkey.ra,dyn.trq_range[i][0],dyn.trq_range[i][1])     # Torque limits
                elif i==50:                                                 # The bound on error i.e t
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                elif i>=20 and i<=43 and (i-1)%3==0:                        # Z-Ground reaction force>=0
                    task.putvarbound(i,mosek.boundkey.lo,0,+inf)
                else:                                                       # Otherwise
                    task.putvarbound(i,mosek.boundkey.fr,-inf,+inf)

            ## Equality Constraints for Dynamics

            numcon=0                        # Keep track of the number of constraints already used
            numcon_eq=6                     # Number of equality constraints            

            task.appendcons(numcon_eq)

            fx=mosek.boundkey.fx                                        # Setting up boundkey for equality constraint i.e HX=B

            for i in range(0+numcon,6+numcon):
                task.putconbound(i,fx,dyn.b_t[i][0],dyn.b_t[i][0])      # Setting upper=lower bound for equality

            task.putaijlist(dyn.H.x_index,dyn.H.y_index,dyn.H.value)    # Setting the Affine matrix                               

            numcon+=numcon_eq 

            ### Inequality Constraints 

            ## 1) Max,Min bound on e to be t  i.e  inf-norm(e)<=t

            numcon_infnorm=12       # 6 for lower and 6 for upper bounds
            
            task.appendcons(numcon_infnorm)

            up=mosek.boundkey.up    # Bound key- Upper bound 
            
            A_infnorm=np.block([    [np.zeros((6,44)),np.eye(6),-1*np.ones((6,1))],
                                    [np.zeros((6,44)),-1*np.eye(6),-1*np.ones((6,1))]])

            for i in range(0+numcon,12+numcon):        
                task.putconbound(i,up,-inf,0)       # Setting upper bound as zero

            subi,subj,value=get_putaijlist(A_infnorm)                           # Sparse kind of form in H
            task.putaijlist(subi,subj,value)
            
            numcon+=numcon_infnorm


            ## 2) Friction Cone Constraints : 
            #    Used a conservative bound - Restrict Cone to a Pyramid  |F_x+F_y|<F_z/(2*myu) 
            
            myu=0.7                                         ######Friction Coefficient #######

            numcon_fr=16                                    # 8 for upper and 8 for lower bound type constraint 
            task.appendcons(numcon_fr)
            block1=np.array([1, 1, -1/(2*myu)])
            block2=np.array([-1, -1, -1/(2*myu)])

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
            task.__del__
            if (solsta == mosek.solsta.optimal):
                #xx = [0.] * 20
                task.getxxslice(mosek.soltype.bas,0,20,torque) # storing just the torques
                #torque[:]=xx[:]
                #print('yay')
            '''                    
            elif (solsta == mosek.solsta.dual_infeas_cer or solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

            '''
            
        env.__del__
    
    return torque                 




