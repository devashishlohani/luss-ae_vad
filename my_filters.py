from typing import Tuple
import torch
import torch.nn as nn
from kornia.filters.kernels import get_gaussian_kernel2d
from kornia.filters.blur_pool import _blur_pool_by_kernel2d

class GaussPool2D(nn.Module):

    def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float], stride: int = 2):
        super(GaussPool2D, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.stride = stride
        self.register_buffer('kernel', get_gaussian_kernel2d(kernel_size, sigma)) #5x5 kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _blur_pool_by_kernel2d(input, kernel.repeat((input.size(1), 1, 1, 1)), self.stride)