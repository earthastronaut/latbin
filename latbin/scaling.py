#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy
import pandas as pd
from latbin.lattice import ALattice, ZLattice, DLattice
import matplotlib.pyplot as plt

# ########################################################################### #

def binning_entropy(data, lattice, bin_cols=None, bin_prefix="q"):
    gb = lattice.bin(data, bin_cols=bin_cols, bin_prefix=bin_prefix)
    #shannon information == -sum(p*ln(p))
    n = len(data)
    probs = gb.size()/float(n)
    entropy = -np.sum(probs*np.log2(probs))
    return entropy

def binning_mutual_information(data, x_cols, y_cols, x_scale, y_scale, lattice_factory=None):
    if lattice_factory is None:
        lattice_factory = ALattice
    npts = len(data)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(np.asarray(data))
    xdata = data[x_cols]
    ydata = data[y_cols]
    
    xlat = lattice_factory(len(x_cols), scale=x_scale)
    xquantize = xlat.quantize(xdata)
        
    ylat = lattice_factory(len(y_cols), scale=y_scale)
    yquantize = ylat.quantize(ydata)
    
    xcount = {}
    ycount = {}
    xycount = {}
    for i in range(len(data)):
        xtup = tuple(xquantize[i])
        ytup = tuple(yquantize[i])
        xcount[xtup] = xcount.get(xtup, 0) + 1
        ycount[ytup] = ycount.get(ytup, 0) + 1
        xycount[(xtup, ytup)] = xycount.get((xtup, ytup), 0) + 1
    
    mut_info = 0.0
    norm = float(npts)
    for xtup, ytup in xycount.keys():
        px = xcount[xtup]/norm
        py = ycount[ytup]/norm
        pxy = xycount[(xtup, ytup)]/norm
        mut_info += pxy*np.log2(pxy/(px*py))
    return mut_info

def get_entropy_spectrum(data, min_scale=0.1, max_scale=10.0, n_steps=100, n_offsets=10, lattice_factory=None):
    if (min_scale <= 0) or (max_scale <= 0):
        raise ValueError("scales must be > 0")
    data = np.asarray(data)
    if lattice_factory is None:
        lattice_factory = ALattice
    scales = np.exp(np.linspace(np.log(min_scale), np.log(max_scale), n_steps))
    ndim = data.shape[1]
    offsets = np.random.random(size=(n_offsets, ndim,))*max_scale*2.0
    entropies = np.zeros(len(scales))
    for offset_idx in range(len(offsets)):
        offset_data = data-offsets[offset_idx]
        for scale_idx, scale in enumerate(scales):
            lat = lattice_factory(ndim, scale=scale)
            entropies[scale_idx] += binning_entropy(offset_data, lat)
    entropies /= float(n_offsets)
    return ScaleEntropySpectrum(scales, entropies)

class ScaleEntropySpectrum:
    
    def __init__(self, scales, entropies):
        self.scales = scales
        self.entropies = entropies
    
    def entropy_derivatives(self):
        d1 = scipy.gradient(self.entropies)
        d2 = scipy.gradient(d1)
        return d1, d2
        
    def plot(self, show=False, label_axes=True):
        d1, d2 = self.entropy_derivatives()
        plt.plot(np.log10(self.scales), d1)
        plt.plot(np.log10(self.scales), d2)
        if show:
            plt.show()

    
