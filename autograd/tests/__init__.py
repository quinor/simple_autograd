import unittest
import numpy as np
import autograd
import autograd.operations as ops


class TestBasicOperators(unittest.TestCase):
    def setUp(self):
        self.a = autograd.Tensor(np.random.random(10), requires_grad=True)
        self.b = autograd.Tensor(np.random.random(10), requires_grad=True)
        self.g = autograd.Tensor(np.random.random(10))

    def test_add(self):
        x = ops.add(self.a, self.b)
        y = self.a.data + self.b.data
        self.assertTrue(np.allclose(x.data, y))

        x.backward(self.g)
        self.assertTrue(np.allclose(self.a.grad.data, self.g.data))
        self.assertTrue(np.allclose(self.b.grad.data, self.g.data))

    def test_mul(self):
        x = ops.mul(self.a, self.b)
        y = self.a.data * self.b.data
        self.assertTrue(np.allclose(x.data, y))

        x.backward(self.g)
        self.assertTrue(np.allclose(self.a.grad.data, self.g.data*self.b.data))
        self.assertTrue(np.allclose(self.b.grad.data, self.g.data*self.a.data))


if __name__ == "__main__":
    unittest.main()
