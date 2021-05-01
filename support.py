import mujoco_py as mp
import numpy as np



def hold_in_air(model,sim):
    # Torso pos,Orientation
    for i in range(0,7,1):
        sim.data.qpos[i]=0
    # Z-height
    sim.data.qpos[2]=0.93 # 0.93
    # Hip Pitch
    sim.data.qpos[7]=np.deg2rad(21.5)       # Left
    sim.data.qpos[34]=np.deg2rad(-21.5)     # Right
    # Yaw,Pitch
    sim.data.qpos[8]=0
    sim.data.qpos[9]=0
    sim.data.qpos[35]=0
    sim.data.qpos[36]=0
    # Rest of the leg
    sim.data.qpos[14:31]=np.zeros((31-14))
    sim.data.qpos[41:55]=np.zeros((55-41))
    # Z- Velocity
    sim.data.qvel[2]=0

    return sim


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

    return P_cm
