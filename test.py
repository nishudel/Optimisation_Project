import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
'''
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

    return pos_tr       #(3x1)
    

    
# Homogenous transformation vector     

# Calculate the jacobian transpose asociated to 
# 4 corner points at each foot
# We get 24 jacobians i.e, one each for x,y,z corrdinates at each point   
# Dimension of foot
l=0.24                                  
w=0.08
# End point coordinates wrt foot frame - Common for both feet
f1=np.array([l/2,-w/2,0])                 #      ***l2***--l1***
f2=np.array([l/2,w/2,0])                  #         |      |
f3=np.array([-l/2,w/2,0])                 #         |      |
f4=np.array([-l/2,-w/2,0])                #      ***l3***--l4**    
ft_ends=[f1,f2,f3,f4]
# Get position of the above points wrt toe roll
# Left foot     - wrt ltr
posl=np.array([0, -0.05456, -0.0315])
eull=np.array([-60, 0, -90])
# Right foot    - wrt rtr
posr=np.array([0, -0.05456, -0.0315])
eulr=np.array([-60, 0, -90])
ft_end_ltr=[]
ft_end_rtr=[]
for elem in ft_ends:
    ft_end_ltr.append(get_pos(eull,posl,elem))
for elem in ft_ends:
    ft_end_rtr.append(get_pos(eull,posl,elem))

for i in ft_end_ltr:
    print(i) 


'''

a=np.ones((4,5))
b=a.reshape((20,1))
print(a)
print(b)