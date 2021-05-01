import mujoco_py as mp
import numpy as np



def hold_in_air(model,sim):
    # Torso pos,Orientation
    for i in range(0,7,1):
        sim.data.qpos[i]=0
    # Z-height
    sim.data.qpos[2]=0.93
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
    sim.data.qpos[41:57]=np.zeros((57-41))
    # Z- Velocity
    sim.data.qvel[2]=0

    return sim