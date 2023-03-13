import numpy as np
from typing import Any
from queue import PriorityQueue
from .tensor import Tensor


class BackwardEngine:
    def __init__(self):
        pass

    def __enter__(self):
        self.gradients = dict()

    def __exit__(self):
        # save gradients to applicable tensors
        for tensor, gradient in self.gradients.items():
            if tensor.requires_grad and tensor.grad_fn is None: # leaf value
                if tensor.grad is None:
                    tensor.grad = gradient
                else:
                    tensor.grad += gradient
        del self.gradients

    def accumulate_grad(self, tensor: Tensor, value: Any):
        if tensor not in self.gradients:
            self.gradients[tensor] = np.zeros_like(tensor.data)

        self.gradients[tensor] += value

    def backward(self, tensor: Tensor, seed_gradient: Any):
        pending = PriorityQueue()
        visited = set()
        self.accumulate_grad(tensor, 1 if seed_gradient is None else seed_gradient)
        if tensor.grad_fn is None:
            return

        pending.put((-tensor.grad_fn.order_no, tensor.grad_fn),)
        set.add(tensor.grad_fn)

        while not pending.empty():
            _, grad_fn = pending.get()

            input_grads = grad_fn.apply_backward(self.gradients[grad_fn.output])
            for input, gradient in zip(grad_fn.inputs, input_grads):
                if input is None:
                    continue
                self.accumulate_grad(input, gradient)
                if input.grad_fn not in visited:
                    pending.put((-grad_fn.order_no, grad_fn),)
                    set.add(grad_fn)
