# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest

# 3rd Party
import numpy as np

# Internal
from latbin import PointInformation
from latbin import ZLattice, DLattice, ALattice, Lattice, generate_lattice

# ########################################################################### #

class TestPointInformation (unittest.TestCase):
    
    def setUp(self):
        
    
        self.pi1 = PointInformation(a=1,b=3,c='hello',e=[1,2,3,4])
        self.pi2 = PointInformation(    b=2,c=' world',e=[5,6])
        
        self.pi3 = PointInformation(a=3.2,b=2.3,c=1.2)
        self.pi4 = PointInformation(      b=5,  c=0, d=-2.3)
        
        unittest.TestCase.setUp(self)   

    def test_add (self):
        
        PI = PointInformation
        
        # these pairs are the (test,answer)
        rpairs = [((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ((self.pi1, self.pi2), PI(a=1,b=5,c='hello world',e=[1,2,3,4,5,6])),
                  #((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  #((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ]
        
        for i,(operands, true_result) in enumerate(rpairs):
            op1, op2 = operands
            add_result = op1.add(op2,fill=0)
            
            msg = " .add failed for example {}".format(i)
            self.assertEqual(add_result, true_result , msg)
        
            dunder_add_result = op1 + op2
            msg = " __add__  failed for example {}".format(i)
            self.assertEqual(dunder_add_result, true_result , msg)
        
    def test_operation(self):
        PI = PointInformation
        rpairs = [((self.pi3, self.pi4), PI(a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ((self.pi1, self.pi2), PI(a=1,b=5,c='hello world',e=[1,2,3,4,5,6])),
                  ]
        
        for i,(operands, true_result) in enumerate(rpairs):
            op1, op2 = operands
            add_result = op1.operation(op2,lambda x,y: x+y,fill=0)
            msg = " .operation failed for example {}".format(i)
            self.assertEqual(add_result, true_result , msg)


            op1.inplace_operation(op2,lambda x,y: x+y,fill=0)
            msg = " .inplace_operation failed for example {}".format(i)
            self.assertEqual(op1, true_result , msg)
    
            try:
                op1.inplace_operation(op2,lambda x: x,fill=0)
            except Exception as e:
                self.assert_(isinstance(e,TypeError), "Takes operators that have args=1")    

class TestLattice (unittest.TestCase):
    
    def setUp (self):
        np.random.seed(89)
        self.rpoints = {}
        for dim in [2, 3, 4]:
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
       
    def test_z1_quantize (self):
        pass
            
    def test_quantize(self):
        pass
      
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
        dimensions = [2, 3, 4]
        lattices = [ALattice, DLattice, ZLattice]
        for dim in dimensions:
            for lat_generator in lattices:
                lat = lat_generator(dim)
                rpoints = self.rpoints[dim]
                quant = lat.quantize(self.rpoints[dim])
                dspace_centers = lat.representation_to_centers(quant)
                import matplotlib.pyplot as plt
                for i in range(len(quant)):
                    plt.plot([rpoints[i, 0], dspace_centers[i, 0]], [rpoints[i, 1], dspace_centers[i, 1]], alpha=0.5)
                q_error = dspace_centers - rpoints
                #plt.hist(q_error[:, 0], 100)
                #plt.hist(q_error[:, 1], 100)
                #plt.scatter(q_error[:, 0], q_error[:, 1])
                plt.show()
                import pdb; pdb.set_trace()
                
        
        
    

# ########################################################################### #

if __name__ == "__main__":
    unittest.main()
