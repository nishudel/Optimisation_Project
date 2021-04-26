import numpy as np
import mujoco_py as mp
from scipy.spatial.transform import Rotation as R


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




