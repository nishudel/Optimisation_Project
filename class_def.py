import numpy as np

class centrl_dyn:
    def __init__(self,H=np.zeros((6,51)),b_t=np.zeros((6,1)),torque=np.zeros(20)):
        self.H=self.get_sparse(H)
        self.b_t=b_t
        self.torque=torque
    class get_sparse:
        def __init__(self,H):
            #non-empty set of index and vals of nparray H
            index=np.transpose(np.nonzero(H))    
            self.x_index=index[:,0]
            self.y_index=index[:,1]
            self.value=H[index[:,0],index[:,1]]
        
class target_dynamics:
    def __init__(self,Kdot=np.zeros((3)),K=np.zeros((3)),kdK=np.ones((3)),Pddot=np.zeros((3)),Pdot=np.zeros((3)),P=np.zeros((3)),kdL=np.ones((3)),kpL=np.ones((3))):
        self.Kdot=Kdot
        self.K=K
        self.kdK=kdK
        self.Pddot=Pddot
        self.Pdot=Pdot
        self.P=P
        self.kdL=kdL
        self.kpL=kpL


