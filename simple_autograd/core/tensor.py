import numpy as np
from .backward_engine import BackwardEngine


class Tensor:
    def __init__(self, value, grad_fn=None):
        self.data = np.asarray(value, dtype=float)
        self.requires_grad = grad_fn is not None
        self.grad = None
        self.grad_fn = None

    def __repr__(self):
        if self.requires_grad:
            return f"Tensor({self.data}, grad_fn={self.grad_fn})"
        else:
            return f"Tensor({self.data})"

    def backward(self, gradient=None):
        with BackwardEngine() as engine:
            engine.backward(self, gradient)


