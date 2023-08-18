import numpy as np
import torch.nn as nn
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.resize import Resize


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """

    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        # self.noise_layers = [Identity()]
        self.noise_layers = []
        for layer in noise_layers:
            if type(layer) is str:
                if layer == "JpegPlaceholder":
                    self.noise_layers.append(JpegCompression(device))
                elif layer == "QuantizationPlaceholder":
                    self.noise_layers.append(Quantization(device))
                else:
                    raise ValueError(
                        f"Wrong layer placeholder string in Noiser.__init__()."
                        f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead'
                    )
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover, noiser_layer_idx=None, noiser_params=None):
        if noiser_layer_idx is not None:
            random_noise_layer = self.noise_layers[noiser_layer_idx]
        else:
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        # print(
        #     f"on device {encoded_and_cover[0].device}, layer is {random_noise_layer}, noiser_params is {noiser_params}"
        # )
        if isinstance(random_noise_layer, Crop) or isinstance(
            random_noise_layer, Resize
        ):
            return random_noise_layer(encoded_and_cover, noiser_params=noiser_params)
        assert (
            noiser_params is None
        ), f"Expected noiser_params to be None for this layer {random_noise_layer}"
        return random_noise_layer(encoded_and_cover)
