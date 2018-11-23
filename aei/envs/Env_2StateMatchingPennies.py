"""
The 2-state Matching Pennies according to HennesEtAl2010
"""

import numpy as np

class TwoStateMatchingPennies(object):

    def __init__(self):
        self.N = 2
        self.M = 2
        self.Z = 2

    def actionticks(self): 
        return [0, 1], ["A", "B"]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        T1 = np.array([[1.0, 1.0],
                       [0.0, 0.0]])
        T2 = np.array([[0.0, 0.0],
                       [1.0, 1.0]])

        Tsas[0, :, :, 1] = T1
        Tsas[0, :, :, 0] = 1-T1
        Tsas[1, :, :, 0] = T2
        Tsas[1, :, :, 1] = 1-T2
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[1 , 0 ],
                            [0 , 1 ]]
        R[1, 0, :, :, 0] = [[0 , 1 ],
                            [1 , 0 ]]

        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[0 , 1 ],
                            [1 , 0 ]]
        R[1, 1, :, :, 1] = [[1 , 0 ],
                            [0 , 1 ]]

        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R