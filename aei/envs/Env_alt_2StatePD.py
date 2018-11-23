"""
The 2-state PD according to Hilbe
"""

import numpy as np

class TwoStatePD(object):

    def __init__(self, b1, b2, c):
        self.N = 2
        self.M = 2
        self.Z = 2

        self.b1 = b1
        self.b2 = b2
        self.c = c

    def actionticks(self): 
        return [0, 1], ["coop.", "defect."]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        T = np.array([[0, 1],
                      [1, 1]])

        Tsas[0, :, :, 1] = T
        Tsas[0, :, :, 0] = 1-T
        Tsas[1, :, :, 0] = 1-T
        Tsas[1, :, :, 1] = T
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[self.b1-self.c, -self.c ],
                            [self.b1       , 0       ]]
        R[1, 0, :, :, 0] = [[self.b1-self.c, self.b1 ],
                            [-self.c       , 0       ]]

        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[self.b2-self.c, -self.c ],
                            [self.b2       , 0       ]]
        R[1, 1, :, :, 1] = [[self.b2-self.c, self.b2 ],
                            [-self.c       , 0       ]]

        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R