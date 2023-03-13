import numpy as np
from queue import PriorityQueue


class Tensor:
    def __init__(self, value, grad_fn=None, requires_grad=False):
        """
        data: np.Array(dtype=float)
        requires_grad: bool
        grad: Tensor(requires_grad=False)
        grad_fn: Function
        """
        self.data = np.asarray(value, dtype=float)
        self.requires_grad = grad_fn is not None or requires_grad
        self.grad = None
        self.grad_fn = None

    def __repr__(self):
        if self.requires_grad and self.grad_fn is not None:
            return f"Tensor({self.data}, grad_fn={self.grad_fn})"
        elif self.requires_grad:
            return f"Tensor({self.data}, requires_grad={self.requires_grad})"
        else:
            return f"Tensor({self.data})"

    def backward(self, gradient=None):
        with BackwardEngine() as engine:
            engine.backward(self, gradient)


class BackwardEngine:
    def __enter__(self):
        self.gradients = dict()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # save gradients to applicable tensors
        for tensor, gradient in self.gradients.items():
            if tensor.requires_grad and tensor.grad_fn is None: # leaf value
                if tensor.grad is None:
                    tensor.grad = gradient
                else:
                    tensor.grad.data += gradient.data # TODO implement += for tensors?
        del self.gradients

    def accumulate_grad(self, tensor: Tensor, value: Tensor):
        if tensor not in self.gradients:
            self.gradients[tensor] = Tensor(np.zeros_like(tensor.data))

        self.gradients[tensor].data += value.data # TODO implement += for tensors?

    def backward(self, tensor: Tensor, seed_gradient: Tensor | None):
        pending = PriorityQueue()
        visited = set()
        self.accumulate_grad(
            tensor,
            Tensor(np.ones_like(tensor.data)) if seed_gradient is None else seed_gradient
        )
        if tensor.grad_fn is None:
            return

        pending.put((-tensor.grad_fn.order_no, tensor.grad_fn),)
        visited.add(tensor.grad_fn)

        while not pending.empty():
            _, cur_grad_fn = pending.get()

            input_grads = cur_grad_fn.apply_backward(self.gradients[cur_grad_fn.output])
            for input, gradient in zip(cur_grad_fn.inputs, input_grads):
                if input is None:
                    continue
                self.accumulate_grad(input, gradient)
                if input.grad_fn is not None and input.grad_fn not in visited:
                    pending.put((-input.grad_fn.order_no, input.grad_fn),)
                    visited.add(input.grad_fn)
