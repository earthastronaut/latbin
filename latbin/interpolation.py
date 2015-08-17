from copy import copy

import numpy as np
import scipy.sparse
import pandas as pd
from latbin.lattice import *
from latbin.matching import MatchingIndexer


class KernelWeightedMatchingInterpolator(object):
    
    def __init__(self, x, y, x_scale, weighting_kernel=None, match_tolerance=6.0):
        if weighting_kernel is None:
            weighting_kernel = lambda x: np.exp(-0.5*x**2)/(1.0+20.0*x**2)
        self.weighting_kernel = weighting_kernel
        self.x_scale = x_scale
        self.m_indexer = MatchingIndexer(x/self.x_scale, tolerance=match_tolerance)
        self.y = y
    
    
    def __call__(self, x_interp):
        x_interp = x_interp/self.x_scale
        dmat = self.m_indexer.distance_matrix(x_interp)
        import matplotlib.pyplot as plt
        import pdb; pdb.set_trace()
        dmat.data = self.weighting_kernel(dmat.data)
        
        weighted_data = dmat*self.y
        weight_sums = dmat*np.ones(len(self.y))
        
        if len(weighted_data.shape) == 1:
            return weighted_data/weight_sums
        else:
            return weighted_data/weight_sums.reshape((-1, 1))

