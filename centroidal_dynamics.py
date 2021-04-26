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

    X_1_G_T=np.block([      [R_1_0              ,np.matmul(R_1_0,np.transpose(skew(P_b_cm)))],
                            [np.zeros((3,3))    ,R_1_0 ]])

    return X_1_G_T      #(6x6)

# Isolating Base frame quantities
def get_U1(model):
    U1= np.block([np.ones((6,6)) ,np.zeros((6,model.nv-6))])
    return U1       #(6xnv)   


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

    return AHinv        #(6xnv)

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
    
    jac_p=np.transpose(np.reshape(jacp,(3,48)))

    G=np.zeros((48,1))
    G[:,0]=jac_p[:,2]
   
    return G  #(nvx1)  

'''
# Mtrix that maps input torque to q
def get_B(model,sim):
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
    
    B=np.block([        [blockb],
                        [blockll],
                        [blockrl],
                        [blocka]    ])

    return B        #(nvx20)



'''

def get_B():
    B=np.zeros((48,20))
    
    row_addr=[0,1,2,6,9,13]     # Non-zero row address for left,right legs
    
    # Legs
    j=0                         # Actuator number
    for i in row_addr:
        B[6+i][j]=1             # Left Leg  6 => Base
        B[23+i][j+6]=1          # Right Leg 23=> Base + Left Leg 
        j=j+1

    j=12                        # Legs make up 12 actuator

    # Arms
    for i in range(0,8):
        B[40+i][j]=1            # Left arm followed by right arm 40 => Base + Both Legs
        j=j+1

    return B                    # (nvx20)



# Vector of holonomic constraint forces (excluding ground contact)
# These are due to the equality constraints imposed 
# at the end of rods 
def get_fhol(model,sim):
    qfrc_force=sim.data.qfrc_constraint
    f_hol=np.zeros((model.nv,1))
    f_hol[:,0]=qfrc_force
    return f_hol    #(nvx1)


# To get the position of a point (vec- known wrt say a "frame")
# wrt the "parent frame" to the "frame"

def get_pos(eul,pos,vec):
    # Part 1
    Rot=R.from_euler('xyz',eul,degrees=True)            # Rotation by euler - XYZ order
    Trans=np.reshape(pos,(3,1))                         # Translation by pos

    Rot=Rot.as_matrix()
    Mat=np.block([      [Rot,Trans],                    # Homogenous Transformation matrix 
                        [0,0,0,1]])


    pos_tr=Mat@np.block([   [vec[0]],
                            [vec[1]],
                            [vec[2]],
                            [1]])
    pos_tr=pos_tr[0:3]
    pos_tr=np.transpose(pos_tr)

    return pos_tr       #(1x3)
    

    
# Homogenous transformation vector     

# Calculate the jacobian transpose asociated to 
# 4 corner points at each foot
# We get 24 jacobians i.e, one each for x,y,z corrdinates at each point
def get_JfootT(model,sim):
    
    # Dimension of foot
    l=0.24                                  
    w=0.08

    # End point coordinates wrt foot frame - Common for both feet

    f1=np.array([l/2,-w/2,0])                 #      ***f2***--f1***                ^ x axis
    f2=np.array([l/2,w/2,0])                  #         |      |                    |
    f3=np.array([-l/2,w/2,0])                 #         |      |                    |
    f4=np.array([-l/2,-w/2,0])                #      ***f3***--f4**  y-axis<--------|    

    ft_ends=[f1,f2,f3,f4]

    # Get position of the above points wrt toe roll

    # Left foot     - wrt ltr
    posl=np.array([0, -0.05456, -0.0315])
    eull=np.array([-60, 0, -90])

    # Right foot    - wrt rtr
    posr=np.array([0, -0.05456, -0.0315])
    eulr=np.array([-60, 0, -90])
    
    ft_end_ltr=np.zeros((1,3))
    ft_end_rtr=np.zeros((1,3))

       
    for elem in ft_ends:
        ft_end_ltr=np.vstack((ft_end_ltr,get_pos(eull,posl,elem)))

    
    
    for elem in ft_ends:
        ft_end_rtr=np.vstack((ft_end_rtr,get_pos(eulr,posr,elem)))
    

    ft_ltr =np.array(ft_end_ltr[1:5,:])
    ft_rtr =np.array(ft_end_rtr[1:5,:]) 

    jac_p_temp=np.ones((3*model.nv))
    jac_r=np.ones((3*model.nv))
    jac_p=np.zeros((48,1))

    for i in range(0,4):
        mp.functions.mj_jac(model,sim.data,jac_p_temp,jac_r,ft_ltr[i,:],12)
        jac=np.reshape(jac_p_temp,(3,model.nv))
        jac=np.transpose(jac)
        jac_p=np.block([jac_p,jac])
    
    jac_l=np.array(jac_p[:,1:13])


    #jac_p_temp=np.ones((3*model.nv))
    #jac_r=np.ones((3*model.nv))
    jac_p=np.zeros((48,1))

    for i in range(0,4):
        mp.functions.mj_jac(model,sim.data,jac_p_temp,jac_r,ft_rtr[i,:],26)
        jac=np.reshape(jac_p_temp,(3,model.nv))
        jac=np.transpose(jac)
        jac_p=np.block([jac_p,jac])
    
    jac_r=np.array(jac_p[:,1:13])
    
    return jac_l,jac_r            #(nvx12) and (nvx12)

    
# Calculate the b_t term of the equations
# Take note of the default value of target dynamics
def get_bt(model,sim,rdot_tc=np.zeros((6,1))):
    AHinv=get_A_Hinv(model,sim)
    G=get_G(model,sim)
    f_hol=get_fhol(model,sim)
    bt=rdot_tc+AHinv@(G-f_hol)

    return bt       #(6x1)


# Calculate cross-coupling inverse operational-space (task-space) inertia matrix

def get_lambdaTF(model,sim):
    AHinv=get_A_Hinv(model,sim)
    B=get_B()
    ft_jac_l=np.empty((48,12))
    ft_jac_r=np.empty((48,12))
    ft_jac=np.empty((48,24))

    ft_jac_l,ft_jac_r=get_JfootT(model,sim)
    ft_jac=np.block([ft_jac_l,ft_jac_r])
    
    lambda_T=AHinv@B                       # For torque
    lambda_F=AHinv@ft_jac                  # For Foot contact

    return lambda_T,lambda_F   #(6x20) and (6x24)                            





    
