import numpy as np
import mujoco_py as mp
from scipy.spatial.transform import Rotation as R

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]]) 


# Calculate CMM

# Transpose of Motion Transformation Matrix from base frame to CoM frame
def get_1XGT(model,sim):
    ## Step 1: Calculate Posiiton of base frame wrt CoM frame of Digit
    
    # Store P_CoM of digit wrt ground
    P_cm=np.empty((3,1))

    #Mass of each body
    body_mass=np.empty((38,3))
    body_mass[:,0]=model.body_mass
    body_mass[:,1]=model.body_mass
    body_mass[:,2]=model.body_mass
    
    # Position of CoM of each body
    body_pos =sim.data.xipos

    mi_pi=np.multiply(body_mass,body_pos)

    P_cm=np.sum(mi_pi,axis=0)

    total_mass=np.sum(body_mass[:,0])

    P_cm=P_cm/total_mass

    # P_base wrt ground
    P_base=sim.data.sensordata[0:3]

    # p_base_CoM
    P_b_cm=P_base-P_cm

    ##  Step 2: Obtain the rotation matrix : Transpose of Base frame wrt ground
    
    # Orientation of base wrt ground
    q_base = sim.data.sensordata[3:7]

    # Rotation Matrix
    R_1_0=np.empty(9)
    mp.functions.mju_quat2Mat(R_1_0,q_base)
    R_1_0=np.transpose(R_1_0.reshape((3,3)))

    ## Step 3: Obtain 1XGT

    X_1_G_T=np.array([      [R_1_0              ,np.matmul(R_1_0,np.transpose(skew(P_b_cm)))],
                            [np.zeros((3,3))    ,R_1_0 ]])

    return X_1_G_T

# Isolating Base frame quantities
def get_U1(model):
    U1= np.array([np.ones((6,6)) ,np.zeros((6,model.nv-6))])
    return U1   


# Get Adot*qdot matirx
def get_Adot_qdot(model,sim):

    X_1_G_T=get_1XGT(model,sim)
    U1=get_U1(model)
    C=sim.data.qfrc_bias

    dAdq=X_1_G_T@U1@C

    return dAdq

# Get A*(H_inverse)    
def get_A_Hinv(model,sim):
    
    X_1_G_T=get_1XGT(model,sim)
    U1=get_U1(model)
    AHinv=X_1_G_T@U1

    return AHinv

# Get G matrix
def get_G(model,sim):
    jacp=np.empty((3*model.nv))
    temp=np.empty((3*model.nv))
    jacr=np.empty((1))

    body_ipos=model.body_ipos
    mi=model.body_mass
    bodylist=np.arange(1,37)

    g=9.8
        
    for i in bodylist:        
        mp.functions.mj_jac(model,sim.data,temp,jacr,body_ipos[i,:],i)
        jacp+=mi[i]*g*temp
    
    jac_p=np.reshape(jacp,(3,48))
    print(sim.data.time)
    print(jac_p[2,:])
    
    return jac_p[2,:]

# Mtrix that maps input torque to q
def get_B(sim,model):
    # Base
    blockb=np.block([np.zeros((6,20))])
    # Left leg
    blockll=np.block([  [np.identity(4),            np.zeros((4,16))],
                        [np.zeros((2,20))],
                        [np.zeros((2,4)),               np.identity(2),         np.zeros((2,14))]])

    # Right leg
    blockrl=np.block([  [np.zeros((4,6)),               np.identity(4),      np.zeros((4,10))],
                        [np.zeros((2,20))],
                        [np.zeros((2,10)),              np.identity(2),         np.zeros((2,8))]])

    # Arms
    sub1=np.zeros((8,12))
    sub2=np.identity(8)
    blocka=np.block([sub1,sub2])

    # Right arm
    #blockra=np.array([  [np.zeros((4,16)),              np.diag([1,1,1,1])]])
    B=np.block([        [blockb],
                        [blockll],
                        [blockrl],
                        [blocka]    ])

    return B

# Vector of holonomic constraint forces (excluding ground contact)
# These are due to the equality constraints imposed 
# at the end of rods 
def get_fhol(model,sim):
    f_hol=sim.data.qfrc_constraint
    return f_hol


# To get the position of a point (vec- known wrt say a "frame")
# wrt the "parent frame" to the "frame"

def get_pos(eul,pos,vec):
    # Part 1
    Rot=R.from_euler('xyz',eul,degrees=True)            # Rotation by euler - XYZ order
    Trans=pos                                           # Translation by pos
    Mat1=np.block([  [Rot,pos],                         # Homogenous Transformation matrix 
                    [0,0,0,1]])

    pos_tr=Mat1@np.block([  [vec],
                            [1]])

    return pos_tr
    

    
# Homogenous transformation vector     

# Calculate the jacobian asociate to 
# 4 corner points at each foot
# We get 24 jacobians i.e, one each for x,y,z corrdinates at each point
def get_JfootT(model,sim):
    
    # Dimension of foot
    l=0.24                                  
    w=0.08

    # End point coordinates wrt foot frame - Common for both feet

    l1=np.array([l/2,-w/2,0])                 #      ***l2***--l1***
    l2=np.array([l/2,w/2/0])                  #         |      |
    l3=np.array([-l/2,w/2,0])                 #         |      |
    l4=np.array([-l/2,-w/2,0])                #      ***l3***--l4**    

    ft_ends=[l1,l2,l3,l4]

    # Get position of the above points wrt toe roll

    # Left foot
    posl=np.array([0, -0.05456, -0.0315])
    eull=np.array([-60, 0, -90])

    # Right foot
    posr=np.array([0, -0.05456, -0.0315])
    eulr=np.array([-60, 0, -90])

    ft_end_ltr=[]
    ft_end_rtr=[]

    for elem in ft_ends:
        ft_end_ltr.append(get_pos(eull,posl,elem))

    for elem in ft_ends:
        ft_end_rtr.append(get_pos(eull,posl,elem))        

    ft_jac_l=[] # Left foot Jacobian
    ft_jac_r=[] # Right foot Jacobian
    i_ltr=10    # id-ltr
    i_rtr=20    # id-rtr
    temp=np.array((3*model.nv))
    jacr=np.empty((1))

    # Get the jacobians
    for elem in ft_end_ltr:
        mp.functions.mj_jac(model,sim.data,temp,jacr,elem,i_ltr) ###  ###Check mode of input for elem
        ft_jac_l.append(temp)
    
    for elem in ft_end_rtr:
        mp.functions.mj_jac(model,sim.data,temp,jacr,elem,i_rtr) ###  ###Check mode of input for elem
        ft_jac_r.append(temp)

    
    return ft_end_ltr,ft_jac_r
    

def get_bt(sim,model,rdot_tc=np.zeros((6,1))):
    AHinv=get_A_Hinv(model,sim)
    G=get_G(model,sim)
    bt=rdot_tc+AHinv@G

    return bt





    
