import argparse
from collections import defaultdict
import logging
import os
import pickle
import pprint
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF
from average_meter import AverageMeter

import utils
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from noise_argparser import NoiseArgParser
from noise_layers.crop import Crop, get_random_rectangle_inside, random_float
from noise_layers.crop_scale import CropScale
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.gaussian_noise import GaussianNoise
from noise_layers.jpeg_compress import JPEGCompression
from noise_layers.noiser import Noiser
from noise_layers.resize import Resize
from noise_layers.rotation import Rotation
from options import HiDDenConfiguration, TrainingOptions
from vgg_loss import VGGLoss

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class HiddenLightning(LightningModule):
    def __init__(
        self,
        configuration: HiDDenConfiguration,
        noise_config: list,
        encoder_decoder: EncoderDecoder,
        discriminator: Discriminator,
        tb_logger: Any,
    ):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(HiddenLightning, self).__init__()
        if self.rank == 0:
            logging.info("HiDDeN model: {}\n".format(self.to_stirng()))
        # self._noiser = noiser
        self.encoder_decoder = encoder_decoder
        self.discriminator = discriminator

        # self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        # self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        self.train_loader = train_loader
        self.val_loader = val_loader
        ## TODO: Add Watson-VGG loss from Stable Signature
        # if configuration.use_vgg:
        #     self.vgg_loss = VGGLoss(3, 1, False)
        #     self.vgg_loss.to(device)
        # else:
        #     self.vgg_loss = None

        self.config = configuration
        self.rank = rank
        # self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

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

    def training_step(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        optimizer_enc_dec, optimizer_discrim = self.optimizers()

        # self.encoder_decoder.train()
        # self.discriminator.train()
        # with torch.enable_grad():
        # ---------------- Train the discriminator -----------------------------
        self.toggle_optimizer(optimizer_discrim)

        optimizer_discrim.zero_grad()
        # train on cover
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
        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
        self.manual_backward(d_loss_on_cover)

        # d_loss_on_cover.mean().backward()
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

    def configure_optimizers(self) -> Any:
        optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())
        return [optimizer_enc_dec, optimizer_discrim], []


def parse_args():
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parent_parser = argparse.ArgumentParser(description="Training of HiDDeN nets")
    subparsers = parent_parser.add_subparsers(
        dest="command", help="Sub-parser for commands"
    )
    new_run_parser = subparsers.add_parser("new", help="starts a new run")
    new_run_parser.add_argument(
        "--data-dir",
        "-d",
        required=True,
        type=str,
        help="The directory where the data is stored.",
    )
    new_run_parser.add_argument(
        "--batch-size", "-b", required=True, type=int, help="The batch size."
    )
    new_run_parser.add_argument(
        "--epochs",
        "-e",
        default=300,
        type=int,
        help="Number of epochs to run the simulation.",
    )
    new_run_parser.add_argument(
        "--name", required=True, type=str, help="The name of the experiment."
    )
    new_run_parser.add_argument(
        "--runs_folder",
        required=True,
        type=str,
        help="The folder to store experiment results.",
    )

    new_run_parser.add_argument(
        "--size",
        "-s",
        default=128,
        type=int,
        help="The size of the images (images are square so this is height and width).",
    )
    new_run_parser.add_argument(
        "--message",
        "-m",
        default=30,
        type=int,
        help="The length in bits of the watermark.",
    )
    new_run_parser.add_argument(
        "--continue-from-folder",
        "-c",
        default="",
        type=str,
        help="The folder from where to continue a previous run. Leave blank if you are starting a new experiment.",
    )
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    new_run_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Use to switch on Tensorboard logging.",
    )
    new_run_parser.add_argument(
        "--enable-fp16",
        dest="enable_fp16",
        action="store_true",
        help="Enable mixed-precision training.",
    )

    new_run_parser.add_argument("--noise_same_on_batch", action="store_true")
    new_run_parser.add_argument(
        "--noise",
        nargs="*",
        action=NoiseArgParser,
        help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'",
    )

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser("continue", help="Continue a previous run")
    continue_parser.add_argument(
        "--folder",
        "-f",
        required=True,
        type=str,
        help="Continue from the last checkpoint in this folder.",
    )
    continue_parser.add_argument(
        "--data-dir",
        "-d",
        required=False,
        type=str,
        help="The directory where the data is stored. Specify a value only if you want to override the previous value.",
    )
    continue_parser.add_argument(
        "--epochs",
        "-e",
        required=False,
        type=int,
        help="Number of epochs to run the simulation. Specify a value only if you want to override the previous value.",
    )
    # continue_parser.add_argument('--tensorboard', action='store_true',
    #                             help='Override the previous setting regarding tensorboard logging.')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == "continue":
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, "options-and-config.pickle")
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
            os.path.join(this_run_folder, "checkpoints")
        )
        train_options.start_epoch = checkpoint["epoch"] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, "train")
            train_options.validation_folder = os.path.join(args.data_dir, "val")
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(
                    f"Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} "
                    f"already contains checkpoint for epoch = {train_options.start_epoch}."
                )
                exit(1)

    else:
        assert args.command == "new"
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, "train"),
            validation_folder=os.path.join(args.data_dir, "val"),
            # runs_folder=os.path.join(".", "runs"),
            runs_folder=os.path.join(args.runs_folder, "runs"),
            start_epoch=start_epoch,
            experiment_name=args.name,
        )

        noise_config = args.noise if args.noise is not None else []
        # import ipdb

        # ipdb.set_trace()
        hidden_config = HiDDenConfiguration(
            H=args.size,
            W=args.size,
            message_length=args.message,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=args.enable_fp16,
        )

        this_run_folder = utils.create_folder_for_run(
            train_options.runs_folder, args.name
        )
        with open(
            os.path.join(this_run_folder, "options-and-config.pickle"), "wb+"
        ) as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(this_run_folder, f"{train_options.experiment_name}.log")
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if (args.command == "new" and args.tensorboard) or (
        args.command == "continue"
        and os.path.isdir(os.path.join(this_run_folder, "tb-logs"))
    ):
        logging.info("Tensorboard is enabled. Creating logger.")
        from tensorboard_logger import TensorBoardLogger

        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, "tb-logs"))
    else:
        tb_logger = None

    # device_ids = list(range(torch.cuda.device_count()))
    # noiser = Noiser(noise_config, device)
    # model = Hidden(hidden_config, device, noiser, tb_logger)
    # import ipdb

    # ipdb.set_trace()
    # TODO: Re-write restore from inside HiDDeN class
    # if args.command == "continue":
    #     # if we are continuing, we have to load the model params
    #     assert checkpoint is not None
    #     logging.info(f"Loading checkpoint from file {loaded_checkpoint_file_name}")
    #     utils.model_from_checkpoint(model, checkpoint)

    return (
        hidden_config,
        train_options,
        noise_config,
        this_run_folder,
        tb_logger,
    )


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_datasets(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    train_dataset, val_dataset = utils.get_data_loaders(
        hidden_config, train_options, return_loader=False
    )
    return train_dataset, val_dataset


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


if __name__ == "__main__":
    (
        hidden_conf,
        train_opts,
        noise_conf,
        run_folder,
        tb_logging,
    ) = parse_args()
