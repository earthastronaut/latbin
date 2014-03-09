# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest

# 3rd Party
import numpy as np

# Internal
from latbin import ZLattice, DLattice, ALattice, Lattice, generate_lattice, CompositeLattice

# ########################################################################### #

class TestLattice (unittest.TestCase):
    
    def setUp (self):
        np.random.seed(89)
        self.rpoints = {}
        self.dimensions = [2, 3, 4, 5]
        for dim in self.dimensions:
            self.rpoints[dim] = 5.0*(np.random.random((5000, dim))-0.5)
        
        #parameters = []
        #self.point_set = latbin.PointSet()
        
        self.result2 = None
        
        self.pad = None
        
        unittest.TestCase.setUp(self)
        
        self.z1 = ZLattice(ndim=1, origin=(-1,), scale=2.1)
        self.z2 = ZLattice(ndim=2, origin=(-1, 0), scale=(2.1,2))
        # TODO: write test for ndim=5
    
    def test_z1_coords(self):
        #lattice coordinate, data coordinate pairs
        rpairs = [([[0],[1]],  [[-1], [1.1]])]
        
        for i, (lc, dc) in enumerate(rpairs):
            output_dc = self.z1.lattice_to_data_space(lc)
            self.assertTrue(np.all(dc==output_dc), msg="bad {}".format(i))
            
            output_lc = self.z1.data_to_lattice_space(dc)
            self.assertTrue(np.all(lc==output_lc), msg="bad {}".format(i))
    
    def test_z2_coords(self):
        #lattice coordinate, data coordinate pairs
        rpairs = [([[0, 0],[1, 3.5]],  [[-1, 0], [1.1, 7.0]])]
        
        for i, (lc, dc) in enumerate(rpairs):
            output_dc = self.z2.lattice_to_data_space(lc)
            self.assertTrue(np.all(dc==output_dc), msg="bad {}".format(i))
            
            output_lc = self.z2.data_to_lattice_space(dc)
            self.assertTrue(np.all(lc==output_lc), msg="bad {}".format(i))
             
    def test_generate_lattice (self):
        rkws = [dict(ndim=1),
                dict(ndim=1)]
         
        rpairs = [(ZLattice(**rkws[0]),generate_lattice(family='z',**rkws[0])),
                  ]
                   
        for i,(l1,l2) in enumerate(rpairs):
            msg = "generate lattice failed to generate on {}".format(i)
            # self.assert_(l1==l2,msg)
            pass
            # TODO: implement test
        
    def test_lattice_data_space (self):
        
        delta = 1e-14
        ndim = 3
        scale = np.random.normal(0,2,ndim)
        scale[scale==0] = 1
        origin = np.random.normal(0,2,ndim)
        
        kws = dict(scale=scale,origin=origin)
        
        test_lattices = [ZLattice(ndim,**kws),DLattice(ndim,**kws)]
        
        nrows = 10
        data = np.random.normal(0,2,nrows*ndim).reshape((nrows,ndim))
        
        for lattice in test_lattices:
            latcoords = lattice.data_to_lattice_space(data)
            datcoords = lattice.lattice_to_data_space(latcoords)
            
            is_equal = np.all(np.abs(data-datcoords)<delta)
            
            msg = ("lattice {} did not correctly convert back"
                   " to data coords").format(lattice,)
            self.assert_(is_equal,msg)
            
    def test_quantize(self):
        dimensions = self.dimensions
        lattices = [ALattice, DLattice, ZLattice]
        for dim in dimensions:
            for lat_generator in lattices:
                lat = lat_generator(dim)
                rpoints = self.rpoints[dim]
                quant = lat.quantize(rpoints)
                dspace_centers = lat.representation_to_centers(quant)
                q_error = dspace_centers - rpoints
                msg="lattice quantization for %s in %d dimensions failed" % (lat, dim)
                self.assertTrue(np.all(np.abs(q_error) < 1.0), msg) 
                

    def test_composite_lattice (self):
        lattices = [ALattice(2),
                    ZLattice(3),
                    DLattice(2)]
        
        column_idx = [[0,3],
                      None,
                      [1,2]]
        comp_lat = CompositeLattice(lattices,column_idx)
        
        

# ########################################################################### #

if __name__ == "__main__":
    unittest.main()