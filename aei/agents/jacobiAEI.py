# this is jacobiAEI.py

import numpy as np
import itertools as it
from detAEI import detAEI

class JacobiAEI(detAEI):
    
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
        self.Theta = self._obtain_OtherThanIJActionsSummationTensor()
        self.OmegaD = self._obtain_OtherAgentsDerivativeSummationTensor()

        



    # =========================================================================
    #   Derivatives
    # =========================================================================
    
    def obtain_dTss(self, X):
        j = 0
        b = 1
        s = 2
        r = 3
        sprim = 4
        k2l = list(range(5, 5+self.N-1))  # other agents
        c2d = list(range(5+self.N-1, 5+self.N-1 + self.N))  # all actions
        e2f = list(range(4+2*self.N, 4+2*self.N + self.N-1))  # all ot. actions

        sumsis = [[k2l[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.OmegaD, [j]+k2l+[b]+c2d+e2f,
                np.eye(self.Z), [s, r],
                self.T, [s]+c2d+[sprim]] + otherX + [[s, sprim, j, r, b]]

        dTss = np.einsum(*args)
        return dTss
    
    def obtain_dRisa(self, X):
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
                self.R, [i, s]+c2d+[sprim],
                self.T, [s]+c2d+[sprim]] + otherX + [[i, s, a, j, r, b]]

        return np.einsum(*args)
    
    def obtain_dRis(self, X):
        i = 0  # agent i
        j = 1  # agent j
        a = 2  # agent i's action a
        b = 3  # agent j's action b
        s = 4  # state s
        r = 5  # state r
        sprim = 6  # next state

        k2l = list(range(7, 7+self.N-1))  # other agents but j
        c2d = list(range(7+self.N-1, 7+self.N-1 + self.N))  # all actions
        e2f = list(range(6+2*self.N, 6+2*self.N + self.N-1))  # all actions but j's

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[k2l[o], r, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.OmegaD, [j]+k2l+[b]+c2d+e2f,
                np.eye(self.Z), [s, r],
                self.R, [i, s]+c2d+[sprim],
                self.T, [s]+c2d+[sprim]] + otherX + [[i, s, j, r, b]]

        return np.einsum(*args)

    def obtain_dVis(self, X):
        Tss = self.obtain_Tss(X)
        Ris = self.obtain_Ris(X)
        dTss = self.obtain_dTss(X)
        dRis = self.obtain_dRis(X)

        M = np.eye(self.Z) - self.gamma*Tss
        dM = - self.gamma * dTss
        invM = np.linalg.inv(M)
        dinvM = np.einsum(-1 * invM, [0, 1],
                          np.einsum(dM, [0, 1, 2, 3, 4],
                                    invM, [1, 5],
                                    [0, 5, 2, 3, 4]), [1, 2, 3, 4, 5],
                         [0, 2, 3, 4, 5]
                         )

        i, s, sp = 0, 1, 2
        j, r, b = 3, 4, 5

        dinvMR = np.einsum(dinvM, [s, sp, j, r, b], Ris, [i, sp],
                           [i, s, j, r, b])
        invMdR = np.einsum(invM, [s, sp], dRis, [i, sp, j, r, b],
                           [i, s, j, r, b])

        return (1-self.gamma) * (dinvMR + invMdR)
    

    def obtain_dQisa(self, X):
        dRisa = self.obtain_dRisa(X)
        dTss = self.obtain_dTss(X)
        dVis = self.obtain_dVis(X)
        Tss = self.obtain_Tss(X)
        Vis = self.obtain_Vis(X)

        i, s, sp, j, r, b = 0, 1, 2, 3, 4, 5
        dTV = np.einsum(dTss, [s, sp, j, r, b],
                        Vis, [i, sp]) +\
            np.einsum(Tss, [s, sp],
                      dVis, [i, sp, j, r, b])


        return (1 - self.gamma) * dRisa +\
            self.gamma * dTV[:, :, np.newaxis, :, :, :]


    # legacy    
    def obtain_dlogXisa(self, X):
        
        dlogXisa  = np.zeros([self.N, self.Z, self.M, self.N, self.Z, self.M])
        for index, _ in np.ndenumerate(dlogXisa):
            i = index[0]
            s = index[1]
            a = index[2]
            j = index[3]
            r = index[4]
            b = index[5]

            if i == j and r == s and a==b:
                x = X[j, r, b]
                if x == 0:
                    dlogXisa[index] = np.inf
                else:
                    dlogXisa[index] = 1/x
                
        return dlogXisa

    # =========================================================================
    #   HELPER
    # =========================================================================

    def _obtain_OtherThanIJActionsSummationTensor(self):
        """For use in Einstein Summation Convention.

        To sum over the other agents and their respective actionsself.
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N],  # agent j
                              [self.N for _ in range(self.N-2)],  # other agnt
                              [self.M],  # action a of agent i
                              [self.M],  # action b of agent j
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-2)]))  # other a's
        Theta = np.zeros(dim.astype(int), int)

        for index, _ in np.ndenumerate(Theta):
            I = index[0]
            J = index[1]
            notIJ = index[2:self.N]
            A = index[self.N]
            B = index[self.N+1]
            allAct = index[self.N+2:2*self.N+2]
            notAB = index[2*self.N+2:]

            if len(np.unique(np.concatenate(([I], [J], notIJ)))) is self.N:
                # all agents indicides are different

                #if A == allAct[I] and B == allAct[J]:
                if A == allAct[I]:
                    # indices of action A and B fit to allAct
                    i = min(I, J)
                    j = max(I, J)
                    cd = allAct[:i] + allAct[i+1:j] + allAct[j+1:]  # other actionss
                    areequal = [cd[k] == notAB[k] for k in range(self.N-2)]
                    if np.all(areequal):
                        #Theta[index] = 1
                        Theta[index] = 2 * (B == allAct[J]) - 1

        return Theta

    def _obtain_OtherAgentsDerivativeSummationTensor(self):
        """For use in Einstein Summation Convention.

        To sum over the other agents and their respective actionsself.
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N for _ in range(self.N-1)],  # other agnt
                              [self.M],  # agent a of agent i
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-1)]))  # other a's
        Omega = np.zeros(dim, int)

        for index, _ in np.ndenumerate(Omega):
            I = index[0]
            notI = index[1:self.N]
            A = index[self.N]
            allA = index[self.N+1:2*self.N+1]
            notA = index[2*self.N+1:]

            if len(np.unique(np.concatenate(([I], notI)))) is self.N:
                # all agents indicides are different

                #if A == allA[I]:
                #     # action of agent i equals some other action
                cd = allA[:I] + allA[I+1:]  # other actionss
                areequal = [cd[k] == notA[k] for k in range(self.N-1)]
                if np.all(areequal):
                    # Omega[index] = 1
                    Omega[index] = 2*(A==allA[I])-1

        return Omega
