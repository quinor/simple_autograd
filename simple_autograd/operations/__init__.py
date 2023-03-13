import numpy as np
from simple_autograd.core import Tensor, Function

class Add(Function):
    def forward(self, a, b):
        return Tensor(a.data + b.data)

    def backward(self, out_grad):
        return out_grad, out_grad

class Multiply(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return Tensor(a.data * b.data)

    def backward(self, out_grad):
        a, b = self.saved_tensors
        return Tensor(out_grad.data*b.data), Tensor(out_grad.data*a.data)
