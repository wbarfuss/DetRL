"""
The 2-state PD according to HennesEtAl2010
"""

import numpy as np

class TwoStatePD(object):

    def __init__(self, transitions_changed=False):
        self.N = 2
        self.M = 2
        self.Z = 2

        self.changeT = transitions_changed

    def actionticks(self): 
        return [0, 1], ["coop.", "defect."]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        if not self.changeT:
            T = np.array([[0.1, 0.9],
                          [0.9, 0.1]])
        else:
            T = np.array([[0.9, 0.1],
                          [0.1, 0.9]])

        Tsas[0, :, :, 1] = T
        Tsas[0, :, :, 0] = 1-T
        Tsas[1, :, :, 0] = T
        Tsas[1, :, :, 1] = 1-T
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[3 , 0 ],
                            [10, 2 ]]
        R[1, 0, :, :, 0] = [[3 , 10],
                            [0 , 2 ]]

        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[4 , 0 ],
                                     [10, 1 ]]
        R[1, 1, :, :, 1] = [[4 , 10],
                            [0 , 1 ]]

        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R