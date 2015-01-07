from copy import copy

import numpy as np
import scipy.sparse
import pandas as pd
from latbin.lattice import *

def brute_match(data1, data2, tolerance=0):
    """a brute force matching function.
    Uses a slow N^2 double for loop algorithm.
    may be faster than using the match algorithm for situations where
    either the number of matches is a significant fraction of N^2 or
    the number of dimensions is comparable to the number of data points
    or for very small matching problems.
    """
    idxs_1, idxs_2, distances = [], [], []
    for i in range(len(data1)):
        for j in range(len(data2)):
            dist = np.sum((data1[i]-data2[j])**2) 
            if dist <= tolerance**2:
                idxs_1.append(i)
                idxs_2.append(j)
                distances.append(np.sqrt(dist))
    return idxs_1, idxs_2, distances

def match(data1, data2, tolerance=0, cols=None):
    """efficiently find all matching rows between two data sets
    
    data1: numpy.ndarray or pandas.DataFrame
      data set to be matched
    data2: numpy.ndarray or pandas.DataFrame
      data set to be matched
    tolerance: float
      maximum distance between rows to be considered a match.
    cols: list
      the indexes of the columns to be used in the matching.
      if None the columns of data1 are assumed
      e.g. to match between data sets using the first and third columns in
      each data set we would use, 
      cols = [0, 2]
      If the columns we wish to match have different indexes in the different
      data sets we can specify a tuple instead of a single index.
      e.g. in order to match the first column in the first data set with the 
      5th column in the second data set and the 4th column in both data sets
      we could use,
      cols = [(0, 4), 3]
    """
    if not isinstance(data1, pd.DataFrame):
        data1 = pd.DataFrame(np.asarray(data1))
    if not isinstance(data2, pd.DataFrame):
        data2 = pd.DataFrame(np.asarray(data2))
    if cols == None:
        cols = data1.columns
    nmatch_cols = len(cols)
    cols1 = []
    cols2 = []
    for col_idx in range(len(cols)):
        ccol = cols[col_idx]
        if isinstance(ccol, tuple):
            c1, c2 = ccol
            cols1.append(c1)
            cols2.append(c2)
        cols1.append(ccol)
        cols2.append(ccol)
    #quantize
    if tolerance <= 0:
        raise NotImplementedError()
    
    if nmatch_cols <= 2:
        scale_ratio = 2.0
    else:
        scale_ratio = 1.1
    scale = tolerance*scale_ratio
    qlat = ALattice(nmatch_cols, scale=scale)
    
    switched = False
    if len(data2) > len(data1):
        switched = True
        temp = data1
        data1 = data2
        data2 = temp
    
    d1vals = data1[cols1].values
    d2vals = data2[cols2].values
    long_pts = qlat.quantize(d1vals)
    short_pts = qlat.quantize(d2vals)
    
    all_trans = [short_pts]
    minimal_vecs = qlat.minimal_vectors()
    shell_set = set([tuple(vec) for vec in minimal_vecs])
    for expand_iter in range(max(0, nmatch_cols-2)):
        last_set = copy(shell_set)
        for lvec in last_set:
            lvec = np.array(lvec)
            for mvec in minimal_vecs:
                new_vec = tuple(lvec + mvec)
                shell_set.add(new_vec)
    neighbor_vecs = np.array(list(shell_set))
    
    for neighbor_vec in neighbor_vecs:
        all_trans.append(short_pts + neighbor_vec)
    
    #make the shifted points into dictionaries
    cdict = {}
    for atrans in all_trans:
        for pvec_idx in range(len(atrans)):
            pvec = atrans[pvec_idx]
            ptup = tuple(pvec)
            dval = cdict.get(ptup)
            if dval is None:
                dval = set()
            dval.add(pvec_idx)
            cdict[ptup] = dval
    
    idxs_1 = []
    idxs_2 = []
    distances = []
    dthresh = tolerance**2
    for long_idx in range(len(long_pts)):
        ltup = tuple(long_pts[long_idx])
        possible_match_set = cdict.get(ltup)
        if not possible_match_set is None:
            #calculate actual distance
            for match_idx in possible_match_set:
                dist = np.sum((d1vals[long_idx] - d2vals[match_idx])**2)
                if dist < dthresh:
                    idxs_1.append(long_idx)
                    idxs_2.append(match_idx)
                    distances.append(np.sqrt(dist))
    
    if switched:
        temp = idxs_1
        idxs_1 = idxs_2
        idxs_2 = temp
    
    idxs_1 = np.asarray(idxs_1)
    idxs_2 = np.asarray(idxs_2)
    distances = np.asarray(distances)
    
    return idxs_1, idxs_2, distances


def sparse_distance_matrix(data, data2=None, max_dist=1.0, rbf=None, cols=None,):
    """matches the rows of input data against themselves and generates a
    sparse n_rows by n_rows matrix with entries at i, j if and only if 
    np.sum((data[i]-data[j])**2) < max_dist**2 
    the entries are determined by the rbf function provided. 
    
    
    parameters
    ----------
    data: numpy.ndarray or pandas.DataFrame
      the data array (n_points, n_dimensions). 
    data2: numpy.ndarray or pandas.DataFrame
      an optional second data array to match against and measure distance to.
      if not specified then the rows in data are matched against themselves.
    max_dist: float
      pairs of points with distances greater than this will have zero entries
      in the resulting sparse matrix
    rbf: function
      a function to take a vector of distances to a vector of matrix entries.
      defaults to exp(-distance**2)
    cols: see latbin.matching.match documentation for more info 
    """
    if data2 is None:
        data2 = data
    if rbf is None:
        rbf = lambda x: np.exp(-x**2)
    idxs_1, idxs_2, distances = match(data, data2, tolerance=max_dist, cols=cols)
    entries = rbf(distances)
    n1 = len(data)
    n2 = len(data2)
    coomat = scipy.sparse.coo_matrix((entries, (idxs_1, idxs_2)), shape=(n1, n2))
    return coomat
    
