# this is aei_jacobian.py

import numpy as np
import itertools as it
from jacobiAEI import JacobiAEI
from detAC import detAC

class JacobiAC(JacobiAEI, detAC):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 alpha,
                 beta,
                 gamma,
                 roundingprec=9):

        JacobiAEI.__init__(self,
            TranstionTensor,
            RewardTensor,
            alpha,
            beta,
            gamma,
            roundingprec)

        detAC.__init__(self,
            TranstionTensor,
            RewardTensor,
            alpha,
            beta,
            gamma,
            roundingprec)

    def jacobi(self, X):
        A = self.obtain_A(X)
        dA = self.obtain_dA(X)
        B = A.sum(axis=2)
        dB = dA.sum(axis=2)

        n = np.newaxis
        JACOisajrb =\
            (dA*B[:,:,n,n,n,n] - 
             A[:,:,:,n,n,n]*dB[:,:,n,:,:,:]) / (B**2)[:,:,n,n,n,n]
        return JACOisajrb

    def obtain_dA(self, X):
        expabTDisa = np.exp(self.alpha*self.beta*self.TDerror(X))

        KronDisajrb = np.einsum(np.eye(2), [0,3],
                                np.eye(2), [1,4],
                                2*np.eye(2)+np.ones(2)-2, [2,5])
        dtruncedTDis = self.obtain_dtruncedTDisa(X)

        dAdX =\
            expabTDisa[:,:,:, np.newaxis, np.newaxis, np.newaxis] *\
            (KronDisajrb + self.alpha*self.beta*\
             X[:,:,:, np.newaxis, np.newaxis, np.newaxis] *\
             dtruncedTDis)
        return dAdX

    def obtain_A(self, X):
        expabTDisa = np.exp(self.alpha*self.beta*self.TDerror(X))
        return X * expabTDisa
  
    def obtain_dtruncedTDisa(self, X):
        dnextVisa = self.obtain_dnextVisa(X)
        dRisa = self.obtain_dRisa(X)

        return (1-self.gamma) * dRisa + self.gamma * dnextVisa

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================
    
    def obtain_dnextVisa(self, X):
        Vis = self.obtain_Vis(X)
        dVis = self.obtain_dVis(X)

        i = 0  # agent i
        j = 1  # agent j
        a = 2  # agent i's action a
        b = 3  # agent j's action b
        s = 4  # state s
        r = 5  # state r
        sprim = 6  # next state

        k2l = list(range(7, 7+self.N-2))  # other agents
        c2d = list(range(7+self.N-2, 7+self.N-2 + self.N))  # all actions
        e2f = list(range(5+2*self.N, 5+2*self.N + self.N-2))  # all other actions
                                                        # (i.e. none from i and j)

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[k2l[o], s, e2f[o]] for o in range(self.N-2)]  # sum inds
        otherX = list(it.chain(*zip((self.N-2)*[X], sumsis)))

        args = [self.Theta, [i]+[j]+k2l+[a]+[b]+c2d+e2f,
                np.eye(self.Z), [s, r], 
                Vis, [i, sprim],
                self.T, [s]+c2d+[sprim]] + otherX + [[i, s, a, j, r, b]]

        dXTV = np.einsum(*args)
        
        # ----
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts
        j, r, b = 2 + 3*self.N, 3 + 3*self.N, 4 + 3*self.N

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                dVis, [i, sprim, j, r, b],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a, j, r, b]]

        XTdV = np.einsum(*args)

        # ----
        return dXTV + XTdV

