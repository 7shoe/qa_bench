from SDP import SDP
from utils import expand
import numpy as np
import math
import unittest

class TestSDP(unittest.TestCase):
    def test_dimension_x_opt(self):
        self.assertEqual(len(SDP(2, 'min', reg_lambda=0).run(np.zeros(4))[1]), 2)
        self.assertEqual(len(SDP(3, 'min', reg_lambda=0).run(np.zeros(1+3+math.comb(3,2)))[1]), 3)
        self.assertEqual(len(SDP(10, 'min', reg_lambda=0).run(np.zeros(1+10+math.comb(10,2)))[1]), 10)
        self.assertEqual(len(SDP(25, 'min', reg_lambda=0).run(np.zeros(1+25+math.comb(25,2)))[1]), 25)
        
    def test_function_value(self):
        self.assertEqual(SDP(3, 'max', reg_lambda=0).run(alpha)[0], 6.0)
        self.assertEqual(SDP(3, 'min', reg_lambda=0).run(alpha)[0], 0.0)
        self.assertEqual(SDP(3, 'max', reg_lambda=0).run(-1*alpha)[0], 0.0)
        self.assertEqual(SDP(3, 'min', reg_lambda=0).run(-1*alpha)[0], -6.0)
        
    def test_penalty(self):
        self.assertEqual(SDP(3, 'min', reg_lambda=20).run(np.ones(1+3+math.comb(3,2)))[0], -53)
        self.assertEqual(SDP(4, 'max', reg_lambda=1).run(np.ones(1+4+math.comb(4,2)))[0], 7)