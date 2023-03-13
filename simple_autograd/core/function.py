import abc
from typing import Any
from .tensor import Tensor


class Function(abc.ABC):
    counter: int = 0
    def __init__(self):
        self.name = self.__class__
        self.order_no = Function.counter
        Function.counter += 1
        self.inputs = []
        self.saved_tensors = None

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        """
        Compute given function. Only positional arguments are allowed. Returns a Tensor.
        """
        pass

    @abc.abstractmethod
    def backward(self, out) -> list[Tensor | None]:
        """
        Backpropagate through a function. Return a gradient for each of the inputs of forward().
        """
        pass

    def save_for_backward(self, *args):
        self.saved_tensors = args

    def apply_backward(self, src_gradient):
        input_grads = self.backward(src_gradient)
        assert len(input_grads) == len(self.inputs)

        for input, gradient in zip(self.inputs, input_grads):
            assert (input is None) < (gradient is None)
        self.saved_tensors = None
        return input_grads

    def register_input(self, input: Tensor | None):
        self.inputs.append(input)

    def register_output(self, output: Tensor):
        self.output = output

    @classmethod
    def apply(cls, *args: tuple[Any]):
        fn = cls()
        out = fn.forward(*args)

        requires_grad = False
        for input in args:
            if isinstance(input, Tensor) and input.requires_grad:
                requires_grad = True
                fn.register_input(input)
            else:
                fn.register_input(None)
        fn.register_output(out)

        out.requires_grad = requires_grad
        out.grad_fn = fn if requires_grad else None

        return out

    def __repr__(self) -> str:
        return f"<{self.name}>"
