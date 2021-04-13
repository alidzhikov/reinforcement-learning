# -*- coding: utf-8 -*-

import numpy as np

class Gridworld: 
    terminal_s = [(0,0), (4,4)]
    
    def __init__(self):
        # left top right down
        self.A = [(0,-1), (-1,0), (0,1), (1,0)]
        
    def reset(self):
        self.S = np.reshape(np.arange(16),(4,4))
        self.s = [1,1]
        return self.S, self.s


    def step(self, s, a):
        self.s = s or self.s
        spi = np.add(self.s, a).astype(int)
        r = -1
        if 0 <= spi[0] < 4 and 0 <= spi[1] < 4:
            self.s = spi
        else:
            self.s = s
        return self.s, r
    
    
    def is_terminal_s(self, spi):
        return spi[0] == 0 and spi[1] == 0 or \
                spi[0] == 3 and spi[1] == 3
