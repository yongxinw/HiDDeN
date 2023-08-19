import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """

    def __init__(self, config: HiDDenConfiguration, noise_config: list):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = Noiser(noise_config)

        self.decoder = Decoder(config)

    def forward(self, image, message, noiser_layer_idx=None, noiser_params=None):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser(
            [encoded_image, image],
            noiser_layer_idx=noiser_layer_idx,
            noiser_params=noiser_params,
        )
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message

    def decode(self, image):
        return self.decoder(image)
