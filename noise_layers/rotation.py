from typing import List, Optional, Tuple, Union
from torch import Tensor
import torch.nn as nn
from kornia.augmentation import RandomRotation


class Rotation(nn.Module):
    """
    Rotates the image by a random angle.
    """

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float], List[float]],
        same_on_batch: Optional[bool] = False,
    ) -> None:
        super(Rotation, self).__init__()
        self.rotation = RandomRotation(
            degrees=degrees, same_on_batch=same_on_batch, p=1
        )

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        noised_and_cover[0] = self.rotation(noised_image)
        return noised_and_cover
