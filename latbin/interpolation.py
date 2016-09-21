from copy import copy

import numpy as np
import scipy.sparse
import pandas as pd
from latbin.lattice import *
from latbin.matching import MatchingIndexer


class KernelWeightedMatchingInterpolator(object):
    
    def __init__(self, x, y, x_scale, weighting_kernel=None, match_tolerance=6.0):
        if weighting_kernel is None:
            weighting_kernel = lambda x: np.exp(-0.5*x**2)#/(1.0+(x/0.1)**2)
        self.weighting_kernel = weighting_kernel
        self.x_scale = x_scale
        self.m_indexer = MatchingIndexer(x/self.x_scale, tolerance=match_tolerance)
        self.y = y
    
    def __call__(self, x_interp, estimate_variance=False):
        x_interp = x_interp/self.x_scale
        dmat = self.m_indexer.distance_matrix(x_interp)
        dmat.data = self.weighting_kernel(dmat.data)
        
        weighted_data = dmat*self.y
        weight_sums = dmat*np.ones(len(self.y))
        
        if len(weighted_data.shape) == 1:
            y_interp = weighted_data/weight_sums
        else:
            y_interp = weighted_data/weight_sums.reshape((-1, 1))
        
        if not estimate_variance:
            return y_interp
        else:
            sq_diff_sums = np.zeros(y_interp.shape)
            dmat_sort = dmat.tocsc().sorted_indices()
            indices = dmat_sort.indices
            indptr = dmat_sort.indptr
            for col_idx in range(len(indptr)-1):
                lbi = indptr[col_idx]
                ubi = indptr[col_idx+1]
                row_indices = indices[lbi:ubi]
                data_weight = dmat_sort.data[lbi:ubi]
                if len(row_indices) > 0:
                    for row_index, dweight in zip(row_indices, data_weight):
                        delta_y_sq = (self.y[col_idx] - y_interp[row_index])**2
                        sq_diff_sums[row_index] += delta_y_sq*dweight
            
            estimated_var = sq_diff_sums/weight_sums
            return y_interp, estimated_var
