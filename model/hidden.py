import numpy as np
import torch
import torch.nn as nn
from noise_layers.cropout import Cropout
from noise_layers.resize import Resize

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from noise_layers.crop import Crop, get_random_rectangle_inside, random_float


class Hidden:
    def __init__(
        self,
        configuration: HiDDenConfiguration,
        device: torch.device,
        noiser: Noiser,
        tb_logger,
    ):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        device_ids = list(range(torch.cuda.device_count()))

        self._noiser = noiser
        self.encoder_decoder = nn.DataParallel(
            EncoderDecoder(configuration, noiser.cuda()).cuda(), device_ids=device_ids
        )
        self.discriminator = nn.DataParallel(
            Discriminator(configuration).cuda(), device_ids=device_ids
        )
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.DataParallel(
            nn.BCEWithLogitsLoss(), device_ids=device_ids
        )
        self.mse_loss = nn.DataParallel(nn.MSELoss(), device_ids=device_ids)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger

            encoder_final = self.encoder_decoder.encoder._modules["final_layer"]
            encoder_final.weight.register_hook(
                tb_logger.grad_hook_by_name("grads/encoder_out")
            )
            decoder_final = self.encoder_decoder.decoder._modules["linear"]
            decoder_final.weight.register_hook(
                tb_logger.grad_hook_by_name("grads/decoder_out")
            )
            discrim_final = self.discriminator._modules["linear"]
            discrim_final.weight.register_hook(
                tb_logger.grad_hook_by_name("grads/discrim_out")
            )

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full(
                (batch_size, 1), self.cover_label, device=self.device
            )
            d_target_label_encoded = torch.full(
                (batch_size, 1), self.encoded_label, device=self.device
            )
            g_target_label_encoded = torch.full(
                (batch_size, 1), self.cover_label, device=self.device
            )

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(
                d_on_cover, d_target_label_cover
            )
            d_loss_on_cover.mean().backward()
            # train on fake
            # np.random.seed(0)
            noise_layer_idx = np.random.randint(0, len(self._noiser.noise_layers))
            _noiser_layer = self._noiser.noise_layers[noise_layer_idx]
            if isinstance(_noiser_layer, Crop):
                noiser_params = get_random_rectangle_inside(
                    image=images,
                    height_ratio_range=_noiser_layer.height_ratio_range,
                    width_ratio_range=_noiser_layer.width_ratio_range,
                )
            elif isinstance(_noiser_layer, Resize):
                noiser_params = random_float(
                    _noiser_layer.resize_ratio_min, _noiser_layer.resize_ratio_max
                )
            else:
                noiser_params = None
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(
                images,
                messages,
                noiser_layer_idx=noise_layer_idx,
                noiser_params=noiser_params,
            )
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(
                d_on_encoded, d_target_label_encoded
            )

            d_loss_on_encoded.mean().backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(
                d_on_encoded_for_enc, g_target_label_encoded
            )

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = (
                self.config.adversarial_loss * g_loss_adv
                + self.config.encoder_loss * g_loss_enc
                + self.config.decoder_loss * g_loss_dec
            )

            g_loss.mean().backward()
            self.optimizer_enc_dec.step()
        # print(g_loss)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(
            np.abs(decoded_rounded - messages.detach().cpu().numpy())
        ) / (batch_size * messages.shape[1])

        losses = {
            "loss           ": g_loss.mean().item(),
            "encoder_mse    ": g_loss_enc.mean().item(),
            "dec_mse        ": g_loss_dec.mean().item(),
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.mean().item(),
            "discr_cover_bce": d_loss_on_cover.mean().item(),
            "discr_encod_bce": d_loss_on_encoded.mean().item(),
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules["final_layer"]
            self.tb_logger.add_tensor("weights/encoder_out", encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules["linear"]
            self.tb_logger.add_tensor("weights/decoder_out", decoder_final.weight)
            discrim_final = self.discriminator._modules["linear"]
            self.tb_logger.add_tensor("weights/discrim_out", discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full(
                (batch_size, 1), self.cover_label, device=self.device
            ).to(torch.float32)
            d_target_label_encoded = torch.full(
                (batch_size, 1), self.encoded_label, device=self.device
            ).to(torch.float32)
            g_target_label_encoded = torch.full(
                (batch_size, 1), self.cover_label, device=self.device
            ).to(torch.float32)

            d_on_cover = self.discriminator(images)

            d_loss_on_cover = self.bce_with_logits_loss(
                d_on_cover, d_target_label_cover
            )

            noise_layer_idx = np.random.randint(0, len(self._noiser.noise_layers))
            _noiser_layer = self._noiser.noise_layers[noise_layer_idx]
            if isinstance(_noiser_layer, Crop):
                noiser_params = get_random_rectangle_inside(
                    image=images,
                    height_ratio_range=_noiser_layer.height_ratio_range,
                    width_ratio_range=_noiser_layer.width_ratio_range,
                )
            elif isinstance(_noiser_layer, Resize):
                noiser_params = random_float(
                    _noiser_layer.resize_ratio_min, _noiser_layer.resize_ratio_max
                )
            else:
                noiser_params = None
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(
                images,
                messages,
                noiser_layer_idx=noise_layer_idx,
                noiser_params=noiser_params,
            )
            # encoded_images, noised_images, decoded_messages = self.encoder_decoder(
            #     images, messages
            # )

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(
                d_on_encoded, d_target_label_encoded
            )

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(
                d_on_encoded_for_enc, g_target_label_encoded
            )

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = (
                self.config.adversarial_loss * g_loss_adv
                + self.config.encoder_loss * g_loss_enc
                + self.config.decoder_loss * g_loss_dec
            )

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(
            np.abs(decoded_rounded - messages.detach().cpu().numpy())
        ) / (batch_size * messages.shape[1])

        losses = {
            "loss           ": g_loss.mean().item(),
            "encoder_mse    ": g_loss_enc.mean().item(),
            "dec_mse        ": g_loss_dec.mean().item(),
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.mean().item(),
            "discr_cover_bce": d_loss_on_cover.mean().item(),
            "discr_encod_bce": d_loss_on_encoded.mean().item(),
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return "{}\n{}".format(str(self.encoder_decoder), str(self.discriminator))
