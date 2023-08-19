from typing import Optional, Tuple, Union
from torch import Tensor
import torch.nn as nn
from kornia.augmentation import RandomResizedCrop


class CropScale(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Optional[Union[Tensor, Tuple[float, float]]] = (0.08, 1.0),
        ratio: Optional[Union[Tensor, Tuple[float, float]]] = (3.0 / 4.0, 4.0 / 3.0),
        same_on_batch: Optional[bool] = False,
        cropping_mode: Optional[str] = "resample",
    ) -> None:
        super(CropScale, self).__init__()
        self.crop_scale = RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
            same_on_batch=same_on_batch,
            p=1,
            cropping_mode=cropping_mode,
        )

    def forward(self, noised_and_cover, noiser_params=None):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        noised_and_cover[0] = self.crop_scale(noised_image)
        return noised_and_cover
