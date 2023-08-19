from typing import Optional, Tuple, Union

import torch.nn as nn
from kornia.augmentation import RandomGaussianNoise
from torch import Tensor


class GaussianNoise(nn.Module):
    def __init__(
        self,
        mean: Optional[float] = 0.0,
        std: Optional[float] = 1.0,
        same_on_batch: Optional[bool] = False,
    ) -> None:
        super(GaussianNoise, self).__init__()
        self.gaussian_noise = RandomGaussianNoise(
            mean=mean, std=std, same_on_batch=same_on_batch, p=1
        )

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        noised_and_cover[0] = self.gaussian_noise(noised_image)
        return noised_and_cover
