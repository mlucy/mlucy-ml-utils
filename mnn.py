import torch
from torch import nn
from torch.nn import functional as F

class LiarsReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, beta=8):
        print('f2')
        ctx.save_for_backward(input_tensor)
        ctx.beta = beta
        return F.relu(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        print('b2')
        for needs_grad, input_tensor in zip(ctx.needs_input_grad, ctx.saved_tensors):
            if not needs_grad:
                return None
            return torch.where(
                input_tensor >= 0,
                grad_output,
                2*F.sigmoid(input_tensor*ctx.beta)*grad_output,
            )

class LiarsReLU(nn.Module):
    def __init__(self, beta=8):
        super().__init__()
        self.beta = 8

    def forward(self, batch):
        return LiarsReLUFunction.apply(batch)
