import unittest
from utils import expand

class TestExpandFunction(unittest.TestCase):
    def test_order_vector_0(self):
        result = expand(np.array([0,0,0]), 0)
        self.assertEqual(result, array([1]))
    
    def test_order_vector_1(self):
        result = expand(np.array([-12,66,8]), 1)
        self.assertEqual(result, array([-12,66,8]))
    
    def test_order_vector_2(self):
        result = expand(np.array([0,0,0]), 1)
        self.assertEqual(result, array([0,0,0,0,0,0]))
        
    def test_order_vector_0_with_intercept(self):
        result = expand(np.array([0,0,0]), 0)
        self.assertEqual(result, array([1]))
    
    def test_order_vector_1_with_intercept(self):
        result = expand(np.array([-12,66,8]), 1)
        self.assertEqual(result, array([1,-12,66,8]))
    
    def test_order_vector_2_with_intercept(self):
        result = expand(np.array([0,0,0]), 1)
        self.assertEqual(result, array([1,0,0,0,0,0,0]))
        
    def test_order_matrix_0(self):
        result = expand(np.array([[0,0,0], [0,0,0]]), 0)
        self.assertEqual(result, array([[1],[1]]))
    
    def test_order_matrix_1(self):
        result = expand(np.array([[-12,66,8], [67, 99, 88]]), 1)
        self.assertEqual(result, array([[-12,66,8], [67, 99, 88]]))
    
    def test_order_matrix_2(self):
        result = expand(np.array([[0,0,0],[0,0,0]]), 1)
        self.assertEqual(result, array([[0,0,0,0,0,0],[0,0,0,0,0,0]]))
        
    def test_order_matrix_0_with_intercept(self):
        result = expand(np.array([0,0,0]), 0)
        self.assertEqual(result, array([[1],[1]]))
    
    def test_order_matrix_1_with_intercept(self):
        result = expand(np.array([[-12,66,8], [67, 99, 88]]), 1)
        self.assertEqual(result, array([[1,-12,66,8], [1,67, 99, 88]]))
    
    def test_order_matrix_2_with_intercept(self):
        result = expand(np.array([[0,0,0], [0,0,0]]), 1)
        self.assertEqual(result, array([[1,0,0,0,0,0,0],[1,0,0,0,0,0,0]]))
        
    def test_order_too_large_vector(self):
        with self.assertRaises(AssertionError):
            expand(np.array([0,0,0]), 4)
    
    def input_dimension_too_large(self):
        with self.assertRaises(NotImplementedError):
            expand(np.array([[[0,0,0], [0,0,0]], [[1,1,1], [1,1,1]]]), 1)

if __name__ == "__main__":
    unittest.main()