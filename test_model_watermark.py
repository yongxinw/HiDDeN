import argparse
import glob
import os
import random
import time
import uuid

import numpy as np
import torch
import torch.nn
import torchvision.transforms.functional as TF
import wandb
from PIL import Image, ImageFilter
from sklearn import metrics
from torchvision import transforms
from tqdm import tqdm

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from options import HiDDenConfiguration


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def get_dataset(args):
    image_paths = sorted(glob.glob(os.path.join(args.dataset_path, "*")))
    return image_paths


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(
            img1.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(
            img2.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img2)

    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width

    if img.shape[1] > width:
        x = np.random.randint(0, img.shape[1] - width)
    else:
        x = 0
    if img.shape[0] > height:
        y = np.random.randint(0, img.shape[0] - height)
    else:
        y = 0
    img = img[y : y + height, x : x + width]
    return img


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    exp_id = str(uuid.uuid4())
    if args.with_tracking:
        wandb.init(
            project="diffusion_watermark_HiDDeN",
            name=f"{args.run_name}",
            tags=["HiDDeN"],
            id=exp_id,
        )
        wandb.config.update(args)
        table = wandb.Table(
            columns=[
                "gen_no_w",
                "gen_w",
                "no_w_metric",
                "w_metric",
                "loop_time",
                "encode_time",
            ]
        )

    output_dir = os.path.join(
        os.path.dirname(args.dataset_path),
        "HiDDeN",
        exp_id,
        f"{args.run_name}-{args.start}-{args.end}-r{args.r_degree}-jpeg_r{args.jpeg_ratio}-cs{args.crop_scale}-cr{args.crop_ratio}-gauss_blur_r{args.gaussian_blur_r}-gauss_std{args.gaussian_std}-br_f{args.brightness_factor}-rand_aug{args.rand_aug}",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print("output_dir:", output_dir)
    # 1. get dataset
    image_paths = get_dataset(args)

    # 2. load model
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device=device)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    # import ipdb

    # ipdb.set_trace()
    # image_pil = Image.open(args.source_image)
    # image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    # image_tensor = TF.to_tensor(image).to(device)
    # image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    # image_tensor.unsqueeze_(0)

    # 3. get watermark message
    message = torch.Tensor(
        np.random.choice([0, 1], (1, hidden_config.message_length))
    ).to(device)

    no_w_metrics = []
    w_metrics = []
    for i in tqdm(range(args.start, args.end)):
        # 4. get image
        loop_start = time.time()
        image_path = image_paths[i]
        image_pil = Image.open(image_path)
        set_random_seed()
        image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
        image_tensor.unsqueeze_(0)

        encode_start = time.time()
        # 5. encode watermark
        losses, (
            encoded_images,
            noised_images,
            decoded_messages,
        ) = hidden_net.validate_on_batch([image_tensor, message])
        # decoded_w = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        encode_end = time.time()
        encode_time = encode_end - encode_start
        loop_time = encode_end - loop_start

        # 5.1. save images
        # utils.save_images(
        #     image_tensor.cpu(),
        #     encoded_images.cpu(),
        #     "test",
        #     output_dir,
        #     # resize_to=(256, 256),
        # )

        # 6. image distortion
        image_tensor_pil = TF.to_pil_image((image_tensor[0].cpu().clamp(-1, 1) + 1) / 2)
        encoded_images_pil = TF.to_pil_image(
            (encoded_images[0].cpu().clamp(-1, 1) + 1) / 2
        )
        # encoded_images_pil.save(
        #     os.path.join(output_dir, os.path.basename(image_paths[i]))
        # )
        image_tensor_pil, encoded_images_pil = image_distortion(
            img1=image_tensor_pil, img2=encoded_images_pil, seed=i, args=args
        )

        # 7. decode watermark
        image_tensor_distorted = TF.to_tensor(image_tensor_pil).to(device)
        image_tensor_distorted = image_tensor_distorted * 2 - 1
        image_tensor_distorted.unsqueeze_(0)

        encoded_images_distorted = TF.to_tensor(encoded_images_pil).to(device)
        encoded_images_distorted = encoded_images_distorted * 2 - 1
        encoded_images_distorted.unsqueeze_(0)

        if hasattr(hidden_net.encoder_decoder, "module"):
            decoded_no_w = hidden_net.encoder_decoder.module.decoder(
                image_tensor_distorted
            )
        else:
            decoded_no_w = hidden_net.encoder_decoder.decoder(image_tensor_distorted)
        decoded_no_w = decoded_no_w.detach().cpu().numpy().round().clip(0, 1)

        if hasattr(hidden_net.encoder_decoder, "module"):
            decoded_w = hidden_net.encoder_decoder.module.decoder(
                encoded_images_distorted
            )
        else:
            decoded_w = hidden_net.encoder_decoder.decoder(encoded_images_distorted)
        decoded_w = decoded_w.detach().cpu().numpy().round().clip(0, 1)

        message_detached = message.detach().cpu().numpy()
        metric_w = -np.mean(np.abs(decoded_w - message_detached))
        metric_no_w = -np.mean(np.abs(decoded_no_w - message_detached))
        w_metrics.append(metric_w)
        no_w_metrics.append(metric_no_w)

        # print("original: {}".format(message_detached))
        # print("decoded_w: {}".format(decoded_w))
        # print("decoded_no_w: {}".format(decoded_no_w))
        # print("error_w: {:.3f}".format(metric_w))
        # print("error_no_w: {:.3f}".format(metric_no_w))

        if args.with_tracking:
            table.add_data(
                wandb.Image(image_tensor_pil),
                wandb.Image(encoded_images_pil),
                metric_no_w,
                metric_w,
                loop_time,
                encode_time,
            )
        else:
            if not os.path.exists(os.path.join(output_dir, "gen_no_w")):
                os.makedirs(os.path.join(output_dir, "gen_no_w"), exist_ok=True)
            image_tensor_pil.save(
                os.path.join(output_dir, "gen_no_w", os.path.basename(image_path))
            )
            if not os.path.exists(os.path.join(output_dir, "gen_w")):
                os.makedirs(os.path.join(output_dir, "gen_w"), exist_ok=True)
            encoded_images_pil.save(
                os.path.join(output_dir, "gen_w", os.path.basename(image_path))
            )

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])
    # compute auc roc
    preds = w_metrics + no_w_metrics
    gt_labels = [1] * len(w_metrics) + [0] * len(no_w_metrics)
    # import ipdb

    # ipdb.set_trace()
    fpr, tpr, thresholds = metrics.roc_curve(gt_labels, preds, pos_label=1)
    auc_roc = metrics.auc(fpr, tpr)
    tpr_at_fpr_0_01 = tpr[np.where(fpr <= 0.01)[0][-1]]
    tpr_at_fpr_0_001 = tpr[np.where(fpr <= 0.001)[0][-1]]
    tpr_at_fpr_0_0001 = tpr[np.where(fpr <= 0.0001)[0][-1]]
    import matplotlib.pyplot as plt

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % auc_roc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(os.path.join(output_dir, "roc.png"))
    roc_pil = Image.open(os.path.join(output_dir, "roc.png"))
    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log(
            {
                "AUC_ROC": auc_roc,
                "TPR@FPR=0.01": tpr_at_fpr_0_01,
                "TPR@FPR=0.001": tpr_at_fpr_0_001,
                "TPR@FPR=0.0001": tpr_at_fpr_0_0001,
                "roc_curve": wandb.Image(roc_pil),
            }
        )
    else:
        print("AUC_ROC:", auc_roc)
        print("TPR@FPR=0.01:", tpr_at_fpr_0_01)
        print("TPR@FPR=0.001:", tpr_at_fpr_0_001)
        print("TPR@FPR=0.0001:", tpr_at_fpr_0_0001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument("--run_name", default="run_HiDDeN_wm", type=str)
    parser.add_argument(
        "--options-file",
        "-o",
        default="options-and-config.pickle",
        type=str,
        help="The file where the simulation options are stored.",
    )
    parser.add_argument(
        "--checkpoint-file", "-c", required=True, type=str, help="Model checkpoint file"
    )
    parser.add_argument(
        "--batch-size", "-b", default=12, type=int, help="The batch size."
    )
    # parser.add_argument(
    #     "--source-image", "-s", required=True, type=str, help="The image to watermark"
    # )

    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="/efs/users/yongxinw/outputs/Quartz-SD2-BMW1-1150K-pv_headshots_fullframe_unsplash_mscocoae5_5-lr1e-5-45000.ckpt_newsubset_10000_seed=123_scale=9.0_size=512_ddim/samples",
        default="/efs/users/yongxinw/outputs/run_sd2_gen_image-coco-0-5000-stabilityai_stable-diffusion-2-1-base/gen_no_w",
    )
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=1, type=int)
    parser.add_argument("--with_tracking", action="store_true")

    # for watermarks
    parser.add_argument("--watermark_size", default=512, type=int)

    # for image distortion
    parser.add_argument("--r_degree", default=None, type=float)
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--crop_scale", default=None, type=float)
    parser.add_argument("--crop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rand_aug", default=0, type=int)

    args = parser.parse_args()
    main(args=args)
