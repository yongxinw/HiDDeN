from typing import Optional, Tuple, Union

import torch.nn as nn
from kornia.augmentation import RandomGaussianBlur
from torch import Tensor


class GaussianBlur(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Optional[Union[Tensor, Tuple[float, float]]] = (0.1, 2.0),
        same_on_batch: Optional[bool] = False,
    ) -> None:
        super(GaussianBlur, self).__init__()
        self.gaussian_blur = RandomGaussianBlur(
            kernel_size=kernel_size,
            sigma=sigma,
            same_on_batch=same_on_batch,
            p=1,
        )

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        noised_and_cover[0] = self.gaussian_blur(noised_image)
        return noised_and_cover
