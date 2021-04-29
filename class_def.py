import numpy as np

class centrl_dyn:
    def __init__(self,nu=48,trq_range=np.zeros((20,2)),H=np.zeros((6,51)),b_t=np.zeros((6,1))):
        self.nu=nu
        self.trq_range=trq_range
        self.H=self.get_sparse(H)
        self.b_t=b_t
    class get_sparse:
        def __init__(self,H):
            #non-empty set of index and vals of nparray H
            index=np.transpose(np.nonzero(H))    
            self.x_index=index[:,0]
            self.y_index=index[:,1]
            self.val=H[index[:,0],index[:,1]]
        
                                