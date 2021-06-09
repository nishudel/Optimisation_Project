import numpy as np
import mujoco_py as mp
from scipy.spatial.transform import Rotation as R
import class_def as cldef

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]]) 


# Calculate CMM

# Body id and name just for reference


# Get the position of COM of Digit wrt each body frame
def get_i_P_G(model,sim):
    # Store P_CoM of digit wrt ground
    p_cm=np.empty((3,1))

    # Mass of each body
    body_mass=np.empty((38,3))
    body_mass[:,0]=model.body_mass
    body_mass[:,1]=model.body_mass
    body_mass[:,2]=model.body_mass

    # Position of CoM of each body
    body_pos =sim.data.xipos

    # Multiply it with the posiiton
    mi_pi=np.multiply(body_mass,body_pos)

    # Calculate the position of COM wrt ground
    p_cm=np.sum(mi_pi,axis=0)
    total_mass=np.sum(body_mass[:,0])
    p_cm=p_cm/total_mass    
    p_x=p_cm[0]*np.ones((38,1))
    p_y=p_cm[1]*np.ones((38,1))
    p_z=p_cm[2]*np.ones((38,1))
    
    P_cm=np.block([p_x,p_y,p_z])
    P_cm=P_cm.reshape((38,3))
    
    # Position of Center of mass wrt body i
    P_cm_i=P_cm-body_pos

    return P_cm_i


# Transpose of Motion Transformation Matrix from base frame to CoM frame
def get_1XGT(model,sim):
    ## Step 1: Calculate Posiiton of base frame wrt CoM frame of Digit 
    P_cm_i=get_i_P_G(model,sim)
    P_cm_b=P_cm_i[0]

    ##  Step 2: Obtain the rotation matrix : Transpose of Base frame wrt ground
    # Orientation of base wrt ground
    q_base = sim.data.sensordata[3:7]

    # Rotation Matrix- Rotate world frame by this to get to base frame 0R1
    R_0_1=np.empty(9)
    mp.functions.mju_quat2Mat(R_0_1,q_base)
    R_0_1=R_0_1.reshape((3,3))

    ## Step 3: Obtain 1XGT

    X_1_G_T=np.block([      [R_0_1              ,np.matmul(R_0_1,np.transpose(skew(P_cm_b)))],
                            [np.zeros((3,3))    ,R_0_1]])

    return X_1_G_T      #(6x6)


# Transpose of Motion Transformation Matrix from base frame i to CoM frame
def get_iXGT(model,sim,i):




# Isolating Base frame quantities
def get_U1(model):
    U1= np.block([np.eye((6)) ,np.zeros((6,model.nv-6))])
    return U1       #(6xnv)   

# Isolating frame i quantities
def get_Ui(model,i):
    Ui=np.zeros((6,model.nv))
    Ui[2][5+i]=1
    return Ui       #(6xnv)  

# Get A*(H_inverse)    
def get_A_Hinv(model,sim):
    X_1_G_T=get_1XGT(model,sim)
    U1=get_U1(model)
    AHinv=X_1_G_T@U1
    
    return AHinv        #(6xnv)


############# Incorrect ############# 
# Get Adot*qdot matirx
def get_Adot_qdot(model,sim):
    X_1_G_T=get_1XGT(model,sim)
    U1=get_U1(model)
    CqG=sim.data.qfrc_bias

    dAdq=X_1_G_T@U1@CqG    # This is actually X_1_G_T@U1@(Cq+G)

    return dAdq
############# Incorrect #############

# Get G matrix
def get_G(model,sim):
    
    body_ipos=model.body_ipos
    mi=model.body_mass
    bodylist=np.arange(1,37)
   
    g=9.8

    jac_p_temp=np.zeros((3*model.nv))
    jac_r=np.zeros((3*model.nv))
    jac_p=np.zeros((model.nv,3))

    for i in bodylist:
        mp.functions.mj_jac(model,sim.data,jac_p_temp,jac_r,body_ipos[i,:],i)
        jac=np.reshape(jac_p_temp,(3,model.nv))
        jac=np.transpose(jac)
        jac_p+=mi[i]*g*jac
        

    G=np.zeros((model.nv,1))
    G[:,0]=-jac_p[:,2]              # G= -dPE/dq
   
    return G  #(nvx1)  

# Vector of holonomic constraint forces (excluding ground contact)
# These are due to the equality constraints imposed 
# at the end of rods 
def get_fhol(model,sim):
    qfrc_force=sim.data.qfrc_constraint
    f_hol=np.zeros((model.nv,1))
    f_hol[:,0]=qfrc_force
    return f_hol    #(nvx1)

# Calculate the b_t term of the equations
# Take note of the default value of target dynamics
def get_bt(model,sim,rdot_tc=np.zeros((6,1))):
    AHinv=get_A_Hinv(model,sim)
    G=get_G(model,sim)
    
    f_hol=get_fhol(model,sim)
    bt=rdot_tc+AHinv@(G-f_hol)
    b_t=np.zeros((6))
    b_t[0:6]=bt[0:6,0]
    return b_t       #(6)

# Troque Mapping
def get_B():
    B=np.zeros((54,20))
    
    row_addr=[0,1,2,6,10,14]     # Non-zero row address for left,right legs
    
    # Legs
    j=0                         # Actuator number
    for i in row_addr:
        B[6+i][j]=1             # Left Leg  6 => Base
        B[30+i][j+10]=1          # Right Leg 26=> Base + Left Leg 
        j=j+1

    # Left Arms
    j=6  
    for i in range(0,4):
        B[26+i][j]=1            # Left arm followed by right arm 46 => Base + Both Legs
        j=j+1

    # Right Arms
    j=16
    for i in range(0,4):
        B[50+i][j]=1            # Left arm followed by right arm 46 => Base + Both Legs
        j=j+1

    return B                    # (nvx20)


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
    jac_p=np.zeros((model.nv,1))

    for i in range(0,4):
        mp.functions.mj_jac(model,sim.data,jac_p_temp,jac_r,ft_ltr[i,:],15)
        jac=np.reshape(jac_p_temp,(3,model.nv))
        jac=np.transpose(jac)
        jac_p=np.block([jac_p,jac])
    
    jac_l=np.array(jac_p[:,1:13])


    #jac_p_temp=np.ones((3*model.nv))
    #jac_r=np.ones((3*model.nv))
    jac_p=np.zeros((model.nv,1))

    for i in range(0,4):
        mp.functions.mj_jac(model,sim.data,jac_p_temp,jac_r,ft_rtr[i,:],33)
        jac=np.reshape(jac_p_temp,(3,model.nv))
        jac=np.transpose(jac)
        jac_p=np.block([jac_p,jac])
    
    jac_r=np.array(jac_p[:,1:13])
    
    return jac_l,jac_r            #(nvx12) and (nvx12)

    



# Calculate cross-coupling inverse operational-space (task-space) inertia matrix

def get_lambdaTF(model,sim):
    AHinv=get_A_Hinv(model,sim)
    B=get_B()
    ft_jac_l=np.empty((model.nv,12))
    ft_jac_r=np.empty((model.nv,12))
    ft_jac=np.empty((model.nv,24))

    ft_jac_l,ft_jac_r=get_JfootT(model,sim)
    ft_jac=np.block([ft_jac_l,ft_jac_r])
    
    lambda_T=AHinv@B                       # For torque
    #print(lambda_T)
    lambda_F=AHinv@ft_jac                  # For Foot contact

    return lambda_T,lambda_F   #(6x20) and (6x24)                            



def get_dynamics(model,sim,rdot_tc):
    ## Dynamics: [ lambda_T lambda_Foot ones zeros]  6x(20 24 6 1)
    lambda_T=np.empty((6,20))
    lambda_F=np.empty((6,24))
    H=np.empty((6,51))

    lambda_T,lambda_F=get_lambdaTF(model,sim)
    H=np.block([lambda_T, lambda_F, np.eye(6) ,np.zeros((6,1))])
    
    ## b_t
    b_t=np.empty((6))
    b_t=get_bt(model,sim,rdot_tc)
    

    #dyn_data=cldef.centrl_dyn(H,b_t)
    #return dyn_data

    return H,b_t
    
    
## Used in upper level feedback control law

def get_CoMdata(model,sim):

    #Mass of each body
    body_mass=np.empty((model.nbody,3))
    body_mass[:,0]=model.body_mass
    body_mass[:,1]=model.body_mass
    body_mass[:,2]=model.body_mass

    # Position of CoM of each body
    body_pos =sim.data.xipos
    # Position of CoM of each body
    body_vel =sim.data.cvel[0:model.nbody,3:6]

    mi_pi=np.multiply(body_mass,body_pos)
    mi_vi=np.multiply(body_mass,body_vel)

    P_cm=np.sum(mi_pi,axis=0)
    V_cm=np.sum(mi_vi,axis=0)

    total_mass=np.sum(body_mass[:,0])

    P_cm=P_cm/total_mass
    V_cm=V_cm/total_mass

    P_cm=np.transpose(P_cm)     #(3x1)
    V_cm=np.transpose(V_cm)     #(3x1)


    return P_cm,V_cm


def get_CAM(model,sim):
    Hqdot=np.zeros((model.nv))
    mp.functions.mj_mulM(model,sim.data,Hqdot,sim.data.qvel)
    X_1_G_T=get_1XGT(model,sim)
    U1=get_U1(model)
    CAM=X_1_G_T@U1@Hqdot

    return CAM

def get_rdot_tc(model,sim,target):
    Kt=get_CAM(model,sim)
    P_cm,V_cm=get_CoMdata(model,sim)    
    K_des=target.Kdot+np.multiply(target.kdK,(target.K-Kt[0:3]))
    total_mass=np.sum(model.body_mass)
    L_des=total_mass*(target.Pddot+np.multiply(target.kdL,(target.Pdot-V_cm))+np.multiply(target.kpL,(target.P-P_cm)))
    K_des=np.reshape(K_des,(3,1))
    L_des=np.reshape(L_des,(3,1))
    rdot_tc=np.block([  [K_des],
                        [L_des]])
    return rdot_tc

     