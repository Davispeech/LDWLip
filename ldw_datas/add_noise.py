import torch
from torch import Tensor
import numbers
from collections.abc import Sequence
import random
import cv2
import numpy as np


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"
        return s


class GaussianNoise(torch.nn.Module):
    def __init__(self, noise_factor=(0.3, 0.6)):
        super().__init__()
        self.noise_min = noise_factor[0]
        self.noise_max = noise_factor[1]

    def forward(self, inputs: Tensor) -> Tensor:
        noisy = inputs + torch.randn_like(inputs) * random.uniform(self.noise_min, self.noise_max)
        #noisy = inputs + torch.randn_like(inputs) * random.uniform(0.01, 0.05)
        noisy = torch.clip_(noisy, 0.0, 1.0)
        return noisy


class Nonedo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 1

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class MotionBlur(torch.nn.Module):
    def __init__(self, degree=12, angle=30):
        super().__init__()
        self.degree = random.randint(0, degree)
        self.angle = random.randint(0, angle)

    def forward(self, inputs: Tensor) -> Tensor:

        return inputs



