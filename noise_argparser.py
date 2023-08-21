import argparse
import re

from noise_layers.crop import Crop
from noise_layers.crop_scale import CropScale
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.gaussian_noise import GaussianNoise
from noise_layers.identity import Identity

# from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg_compress import JPEGCompression
from noise_layers.quantization import Quantization
from noise_layers.resize import Resize
from noise_layers.rotation import Rotation


def parse_pair(match_groups):
    heights = match_groups[0].split(",")
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(",")
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(
        r"crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)", crop_command
    )
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))


def parse_cropout(cropout_command):
    matches = re.match(
        r"cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)",
        cropout_command,
    )
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r"dropout\((\d+\.*\d*,\d+\.*\d*)\)", dropout_command)
    ratios = matches.groups()[0].split(",")
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))


def parse_resize(resize_command):
    matches = re.match(r"resize\((\d+\.*\d*,\d+\.*\d*)\)", resize_command)
    ratios = matches.groups()[0].split(",")
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))


def parse_crop_scale(namespace, crop_scale_command):
    matches = re.match(
        r"crop_scale\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)",
        crop_scale_command,
    )
    scales = matches.groups()[0].split(",")
    min_scale = float(scales[0])
    max_scale = float(scales[1])
    ratios = matches.groups()[1].split(",")
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return CropScale(
        size=[namespace.size, namespace.size],
        scale=(min_scale, max_scale),
        ratio=(min_ratio, max_ratio),
        same_on_batch=namespace.noise_same_on_batch,
    )


def parse_rotation(namespace, rotation_command):
    matches = re.match(r"rotation\((\d+\.*\d*)\)", rotation_command)
    angle = float(matches.groups()[0])
    return Rotation(degrees=angle, same_on_batch=namespace.noise_same_on_batch)


def parse_jpeg_compression(namespace, jpeg_command):
    matches = re.match(r"jpeg_compression\((\d+\.*\d*,\d+\.*\d*)\)", jpeg_command)
    qualities = matches.groups()[0].split(",")
    min_quality = float(qualities[0])
    max_quality = float(qualities[1])
    return JPEGCompression(
        size=(namespace.size, namespace.size), quality_range=(min_quality, max_quality)
    )


def parse_gaussian_blur(namespace, gaussian_blur_command):
    matches = re.match(r"gaussian_blur\((\d+\.*\d*,\d+\.*\d*)\)", gaussian_blur_command)
    sigmas = matches.groups()[0].split(",")
    min_sigma = float(sigmas[0])
    max_sigma = float(sigmas[1])
    return GaussianBlur(
        kernel_size=3,
        sigma=(min_sigma, max_sigma),
        same_on_batch=namespace.noise_same_on_batch,
    )


def parse_gaussian_noise(namespace, gaussian_noise_command):
    matches = re.match(r"gaussian_noise\((\d+\.*\d*)\)", gaussian_noise_command)
    std = float(matches.groups()[0])
    return GaussianNoise(std=std, same_on_batch=namespace.noise_same_on_batch)


class NoiseArgParser(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        argparse.Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values, option_string=None):
        layers = []
        split_commands = values[0].split("+")

        for command in split_commands:
            # remove all whitespace
            command = command.replace(" ", "")
            if command[: len("cropout")] == "cropout":
                layers.append(parse_cropout(command))
            # elif command[: len("crop")] == "crop":
            #     layers.append(parse_crop(command))
            elif command[: len("dropout")] == "dropout":
                layers.append(parse_dropout(command))
            # elif command[: len("resize")] == "resize":
            #     layers.append(parse_resize(command))
            # elif command[: len("jpeg")] == "jpeg":
            #     layers.append("JpegPlaceholder")
            # elif command[: len("quant")] == "quant":
            #     layers.append("QuantizationPlaceholder")
            elif command[: len("crop_scale")] == "crop_scale":
                layers.append(parse_crop_scale(namespace, command))
            elif command[: len("rotation")] == "rotation":
                layers.append(parse_rotation(namespace, command))
            elif command[: len("gaussian_blur")] == "gaussian_blur":
                layers.append(parse_gaussian_blur(namespace, command))
            elif command[: len("gaussian_noise")] == "gaussian_noise":
                layers.append(parse_gaussian_noise(namespace, command))
            elif command[: len("jpeg_compression")] == "jpeg_compression":
                layers.append(parse_jpeg_compression(namespace, command))
            elif command[: len("identity")] == "identity":
                # We are adding one Identity() layer in Noiser anyway
                layers.append(Identity())
            else:
                raise ValueError("Command not recognized: \n{}".format(command))
            # TODO: Add gaussian noise/filter layers
            # TODO: Add rotation layers
        setattr(namespace, self.dest, layers)
