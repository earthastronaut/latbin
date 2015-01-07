# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest
import time

# 3rd Party
import numpy as np

# Internal
from latbin import ZLattice, DLattice, ALattice, Lattice, generate_lattice, CompositeLattice
import latbin.matching as matching

# ########################################################################### #

class TestMatch(unittest.TestCase):
    ndim = 2
    match_dist = 0.025
    
    def setUp(self):
        npts1 = 400
        npts2 = 400
        self.d1 = np.random.random((npts1, self.ndim))
        self.d2 = np.random.random((npts2, self.ndim))
    
    def test_match(self):
        stime = time.time()
        match_res = matching.match(self.d1, self.d2, self.match_dist)
        etime = time.time()
        print("beginning matching algorithm")
        print("{} matches found in {} seconds".format(len(match_res[0]), etime-stime))
        
        print("beginning brute matching algorithm")
        stime = time.time()
        brute_res = matching.brute_match(self.d1, self.d2, self.match_dist)
        etime = time.time()
        print("brute match finished in {} seconds".format(etime-stime))
        
        brute_matches = set(zip(brute_res[0], brute_res[1]))
        match_matches = set(zip(match_res[0], match_res[1]))
        
        nmatches = len(brute_res[0])
        
        invalid = match_matches-brute_matches
        self.assertTrue(invalid == set(), "found {} false matches".format(len(invalid)))
        
        missed = brute_matches-match_matches
        self.assertTrue(missed == set(), "missed {} out of {} valid matches".format(len(missed), nmatches))

class TestMatch3D(TestMatch):
    ndim = 3
    match_dist = 0.05

class TestMatch4D(TestMatch):
    ndim = 4
    match_dist = 0.07

class TestMatch5D(TestMatch):
    ndim = 5
    match_dist = 0.1

class TestMatch6D(TestMatch):
    ndim = 6
    match_dist = 0.15


if __name__ == "__main__":
    unittest.main()
