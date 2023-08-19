import os
import sys

sys.path.append(os.path.realpath("./DiffJPEG"))
sys.path.append(os.path.realpath("./DiffJPEG/modules"))
from DiffJPEG import DiffJPEG
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


class JPEGCompression(nn.Module):
    def __init__(
        self,
        size: Tuple[float, float],
        quality_range: Optional[Tuple[float, float]] = (0.25, 0.8),
        **kwargs
    ) -> None:
        super(JPEGCompression, self).__init__()
        self.quality_range = quality_range
        self.height, self.width = size

    def forward(self, noised_and_cover, noiser_params=None):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        quality = np.random.uniform(self.quality_range[0], self.quality_range[1])
        diff_jpeg = DiffJPEG(
            height=self.height, width=self.width, differentiable=True, quality=quality
        ).to(noised_image.device)
        noised_and_cover[0] = diff_jpeg(noised_image)
        return noised_and_cover
