# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest

# 3rd Party

# Internal
from latbin import PointInformation
from latbin import ZLattice 

# ########################################################################### #

class TestPointInformation (unittest.TestCase):
    
    def setUp(self):
        
        lat = ZLattice(2)
        self.lattice = lat
        self.pi1 = PointInformation(lat,a=1,b=3,c='hello',e=[1,2,3,4])
        self.pi2 = PointInformation(lat,    b=2,c=' world',e=[5,6])
        
        self.pi3 = PointInformation(lat,a=3.2,b=2.3,c=1.2)
        self.pi4 = PointInformation(lat,     b=5,  c=0, d=-2.3)
        
        unittest.TestCase.setUp(self)   

    def test_add (self):
        
        PI = PointInformation
        
        # these pairs are the (test,answer)
        rpairs = [((self.pi3, self.pi4), PI(self.lattice,a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ((self.pi1, self.pi2), PI(self.lattice,a=1,b=5,c='hello world',e=[1,2,3,4,5,6])),
                  #((self.pi3, self.pi4), PI(self.lattice,a=3.2,b=7.3,c=1.2,d=-2.3)),
                  #((self.pi3, self.pi4), PI(self.lattice,a=3.2,b=7.3,c=1.2,d=-2.3)),
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
        rpairs = [((self.pi3, self.pi4), PI(self.lattice,a=3.2,b=7.3,c=1.2,d=-2.3)),
                  ((self.pi1, self.pi2), PI(self.lattice,a=1,b=5,c='hello world',e=[1,2,3,4,5,6])),
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

# ########################################################################### #

if __name__ == "__main__":
    unittest.main()
