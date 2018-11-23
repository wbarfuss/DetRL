"""
This is detSARSA.py, the determinsitic limit of SARSA learning.
"""
import itertools as it
import numpy as np
from detAEI import detAEI

class detSARSA(detAEI):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 alpha,
                 beta,
                 gamma,
                 roundingprec=9):

        detAEI.__init__(self,
                        TranstionTensor,
                        RewardTensor,
                        alpha,
                        beta,
                        gamma,
                        roundingprec)

    # =========================================================================
    #   Temporal difference error
    # =========================================================================

    def TDerror(self, X, norm=True):
        Risa = self.obtain_Risa(X)
        NextQisa = self.obtain_NextQisa(X)
        
        TDe = (1 - self.gamma)*Risa + self.gamma*NextQisa - 1/self.beta * np.ma.log(X)
        if norm:
            TDe = TDe - TDe.mean(axis=2, keepdims=True)
        TDe = TDe.filled(0)
        return TDe

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_NextQisa(self, X):
        """
        For SARSA learning
        """
        Qisa = self.obtain_Qisa(X)

        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))                      # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))    # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum ids
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))
        NextQisa = np.einsum(Qisa, [0, 1, 2],
                             X, [0, 1, 2], [0, 1])
        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                NextQisa, [i, sprim],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a]]

        return np.einsum(*args)
