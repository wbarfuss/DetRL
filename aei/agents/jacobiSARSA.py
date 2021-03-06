# this is jacobiSARSA.py

import numpy as np
import itertools as it
from jacobiAEI import JacobiAEI
from detSARSA import detSARSA

class JacobiSARSA(JacobiAEI, detSARSA):

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

        detSARSA.__init__(self,
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
        expabTDisa = np.exp(self.alpha*self.beta*self.obtain_truncedTDisa(X))
        dtruncedTDis = self.obtain_dtruncedTDisa(X)

        KronDisajrb = np.einsum(np.eye(2), [0,3],
                                np.eye(2), [1,4],
                                2*np.eye(2)+np.ones(2)-2, [2,5])
        n = np.newaxis
        dX = (1-self.alpha) * KronDisajrb * X[:,:,:,n,n,n]**(-self.alpha)

        dAdX =\
            expabTDisa[:,:,:, n, n, n] *\
            (dX + self.alpha*self.beta*\
             X[:,:,:, n, n, n]**(1-self.alpha) * dtruncedTDis)
        return dAdX

    def obtain_A(self, X):
        expabTDisa = np.exp(self.alpha*self.beta*self.obtain_truncedTDisa(X))
        return X**(1-self.alpha) * expabTDisa
  
    def obtain_dtruncedTDisa(self, X):
        dnextQisa = self.obtain_dnextQisa(X)
        dRisa = self.obtain_dRisa(X)

        return (1-self.gamma) * dRisa + self.gamma * dnextQisa

    def obtain_truncedTDisa(self, X):
        Risa = self.obtain_Risa(X)
        NextQisa = self.obtain_NextQisa(X)
        
        tTDe = (1 - self.gamma)*Risa + self.gamma*NextQisa

        return tTDe

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_dXQis(self, X):
        Qisa = self.obtain_Qisa(X)
        dQisa = self.obtain_dQisa(X)

        KronDisajrb = np.einsum(np.eye(2), [0,3],
                                np.eye(2), [1,4],
                                2*np.eye(2)+np.ones(2)-2, [2,5])

        n = np.newaxis
        return (KronDisajrb * Qisa[:, :, :, n, n, n]
                + X[:, :, :, n, n, n] * dQisa).sum(axis=2)

    
    def obtain_dnextQisa(self, X):
        Qisa = self.obtain_Qisa(X)
        dXQis = self.obtain_dXQis(X)

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
                (X*Qisa).sum(axis=2), [i, sprim],
                self.T, [s]+c2d+[sprim]] + otherX + [[i, s, a, j, r, b]]

        dXTQ = np.einsum(*args)
        
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
                dXQis, [i, sprim, j, r, b],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a, j, r, b]]

        XTdQ = np.einsum(*args)

        # ----
        return dXTQ + XTdQ

