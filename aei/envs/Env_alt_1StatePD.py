"""
The 1-state PD according to Hilbe
"""

import numpy as np

class OneStatePD(object):

    def __init__(self, b, c):
        self.N = 2
        self.M = 2
        self.Z = 1

        self.b = b
        self.c = c

    def actionticks(self): 
        return [0, 1], ["coop.", "defect."]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((1, 2, 2, 1))
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 1, 2, 2, 1))

        R[0, 0, :, :, 0] = [[self.b-self.c, -self.c ],
                            [self.b       , 0       ]]
        R[1, 0, :, :, 0] = [[self.b-self.c, self.b ],
                            [-self.c       , 0       ]]

        return R