"""
© All rights reserved. EPFL (École Polytechnique Fédérale de Lausanne),
Switzerland, Image and Visual Representation Lab., 2024. Largo.ai, Switzerland,
2024. File created by Martin Nicolas Everaert.

License: only for academic non-commercial usage. Details in the ``LICENSE'' file
(https://github.com/ivrl/signal-leak-bias/blob/main/LICENSE). Please contact 
Largo.ai (`info@largo.ai`) and EPFL-TTO (`info.tto@epfl.ch`) for a full
commercial license.

This file is meant to demonstrate the signal-leak bias presented the paper:

"Exploiting the Signal-Leak Bias in Diffusion Models", Martin Nicolas Everaert,
Athanasios Fitsios, Marco Bocchio, Sami Arpa, Sabine Süsstrunk, Radhakrishna
Achanta. Proceedings of the IEEE/CVF Winter Conference on Applications of 
Computer Vision (WACV), 2024.
"""

import argparse
import glob
import os
from typing import Tuple, List

import numpy as np
import torch
from torchvision import transforms
import tqdm
from diffusers.models import AutoencoderKL
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.fftpack import dct, idct
from torchvision.transforms import Compose


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to compute statistics for the signal leak."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the image directory.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the statistics will be written.",
    )

    parser.add_argument(
        "--statistic_type",
        type=str,
        default="pixel",
        required=True,
        help="How to model the signal leak and compute statistics.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducibility."
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution."
        ),
    )

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally.",
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help=(
            "A number of components used to model the signal leak, depending on how the signal leak is modeled."
            "For example, number of components to model in the frequency domain, the remaining being in the pixel domain."
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    assert os.path.exists(args.data_dir), f"{args.data_dir} does not exist"
    assert not os.path.exists(args.output_dir), f"{args.output_dir} already exists"

    return args


def sample_from_stats(
    path: str,
    dims: Tuple[int, int, int, int] = None,
    generator_pt: torch.Generator = None,
    generator_np: np.random.Generator = None,
    device: torch.device = "cpu",
    only_hf: bool = False,
) -> torch.Tensor:
    """
    Samples a tensor from computed statistics stored in the given path.

    Args:
        path (str): The path to the directory containing the computed statistics.
        dims (Tuple[int, int, int, int]): The dimensions of the tensor to be sampled.
        generator_pt (torch.Generator): PyTorch random number generator.
        generator_np (np.random.Generator): NumPy random number generator.
        device (torch.device): The device for the tensor returned.
        only_hf (bool): Ignore the LF terms and sample only the HF terms

    Returns:
        torch.Tensor: The sampled tensor.

    Raises:
        AssertionError: If the statistic_type is not one of ["dct+pixel", "pixel"].
    """

    type_file = open(f"{path}/statistic_type.txt", "r")
    statistic_type = type_file.read()
    type_file.close()

    assert statistic_type in ["dct+pixel", "pixel"]

    if statistic_type == "dct+pixel":
        # Load computed stats from file
        tensors = {}
        with safe_open(f"{path}/stats.safetensors", framework="pt", device="cpu") as f:
            for k in [
                "mean_lf_components",
                "mean_hf_remainder",
                "cov_lf_components",
                "var_hf_remainder",
            ]:
                tensors[k] = f.get_tensor(k)
        mean_lf_components = tensors["mean_lf_components"]
        mean_hf_remainder = tensors["mean_hf_remainder"]
        cov_lf_components = tensors["cov_lf_components"]
        var_hf_remainder = tensors["var_hf_remainder"]

        if dims is None:
            dims = (1,) + (mean_hf_remainder.size())

        # Sample latents from stats
        if generator_np is None:
            generator_np = np.random.default_rng()
        sample = sample_from_dct_px_stats(
            mean_lf_components,
            mean_hf_remainder,
            cov_lf_components,
            var_hf_remainder,
            generator_pt,
            generator_np,
            dims,
            device,
            only_hf,
        )

    elif statistic_type == "pixel":
        # Load computed stats from file
        tensors = {}
        with safe_open(f"{path}/stats.safetensors", framework="pt", device="cpu") as f:
            for k in ["mean_pixels", "var_pixels"]:
                tensors[k] = f.get_tensor(k)
        mean_pixels = tensors["mean_pixels"]
        var_pixels = tensors["var_pixels"]

        # Sample latents from stats
        sample = sample_from_px_stats(
            mean_pixels, var_pixels, generator_pt, dims, device
        )

    return sample


def sample_from_dct_px_stats(
    mean_lf_components: torch.Tensor,
    mean_hf_remainder: torch.Tensor,
    cov_lf_components: torch.Tensor,
    var_hf_remainder: torch.Tensor,
    generator_pt: torch.Generator,
    generator_np: np.random.Generator,
    dims: Tuple[int, int, int, int],
    device: torch.device,
    only_hf: bool = False,
) -> torch.Tensor:
    """
    Samples from the low-frequency and high-frequency components of the latents
    according to the given statistics.

    Args:
        mean_lf_components (torch.Tensor): Mean of the low-frequency components.
        mean_hf_remainder (torch.Tensor): Mean of the high-frequency remainder.
        cov_lf_components (torch.Tensor): Covariance matrix of the low-frequency components.
        var_hf_remainder (torch.Tensor): Variance of the high-frequency remainder.
        generator_pt (torch.Generator): PyTorch random number generator.
        generator_np (np.random.Generator): NumPy random number generator.
        dims (Tuple[int, int, int, int]): Dimensions of the tensors.
        device (str): Device to use for computation.

    Returns:
        torch.Tensor: Sampled tensor from the low-frequency and high-frequency components.
    """
    if only_hf:
        lf = 0
    else:
        batch_size, latent_channels, _, _ = dims
        n_components = mean_lf_components.numel() // latent_channels

        # Sample the low-frequency components of the latents according to computed statistics mean_lf_components and cov_lf_components

        x, y = np.meshgrid(np.arange(dims[-2]), np.arange(dims[-2]))
        k = np.sqrt(x**2 + y**2)

        # Get indices (DCT coordinates) of the n_components with lowest frequency -> top left quarter circle
        ind_x, ind_y = np.unravel_index(
            np.argsort(k, axis=None)[:n_components], k.shape
        )

        lf_components = generator_np.multivariate_normal(
            mean_lf_components.numpy(), cov_lf_components.numpy(), batch_size
        )
        lf_components = lf_components.reshape(
            (batch_size, n_components, latent_channels)
        )
        lf_components = lf_components.swapaxes(
            1, -1
        )  # batch_size, latent_channels, n_components

        dct_lf = np.zeros(dims)
        dct_lf[:, :, ind_x, ind_y] = lf_components
        lf = idct2(dct_lf)

        lf = torch.from_numpy(lf)
        lf = lf.to(device=device, dtype=torch.float32)

    mean = mean_hf_remainder[None, :, :, :].to(device=device, dtype=torch.float32)
    std = torch.sqrt(
        var_hf_remainder[None, :, :, :].to(device=device, dtype=torch.float32)
    )
    hf_remainder = mean + std * torch.randn(
        dims,
        generator=generator_pt,
        device=device,
        dtype=torch.float32,
    )

    sample = lf + hf_remainder
    return sample


def sample_from_px_stats(
    mean_pixels: torch.Tensor,
    var_pixels: torch.Tensor,
    generator: torch.Generator,
    dims: Tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a sample tensor from pixel statistics.

    Args:
        mean_pixels (torch.Tensor): Tensor of mean pixel values.
        var_pixels (torch.Tensor): Tensor of variance of pixel values.
        generator (torch.Generator): Random number generator.
        dims (Tuple[int, int, int, int]): Tuple of batch size, latent channels, resolution, and resolution_.
        device (torch.device): Device to place the tensors on.

    Returns:
        torch.Tensor: Generated sample tensor.
    """

    batch_size, latent_channels, resolution, resolution_ = dims
    mean = mean_pixels[None, :, :, :].to(device=device, dtype=torch.float32)
    std = torch.sqrt(var_pixels[None, :, :, :].to(device=device, dtype=torch.float32))
    sample = mean + std * torch.randn(
        (batch_size, latent_channels, resolution, resolution_),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return sample


# Some utility functions for DCT and image loading


def dct2(a: np.ndarray) -> np.ndarray:
    """
    Perform a 2D Discrete Cosine Transform (DCT) on the input array.

    Parameters:
        a (np.ndarray): Input array.

    Returns:
        np.ndarray: Transformed array.
    """
    return dct(dct(a, axis=-2, norm="ortho"), axis=-1, norm="ortho")


def idct2(a: np.ndarray) -> np.ndarray:
    """
    Perform 2D inverse discrete cosine transform (IDCT) on the input array.

    Parameters:
        a (np.ndarray): Input array to perform IDCT on.

    Returns:
        np.ndarray: Output array after performing IDCT.

    """
    return idct(idct(a, axis=-2, norm="ortho"), axis=-1, norm="ortho")


def img_path_to_latents(
    path: str, train_transforms: Compose, vae: AutoencoderKL
) -> torch.Tensor:
    """
    Converts an image file to its corresponding latent representation using a VAE model.

    Args:
        path (str): The path to the image file.
        train_transforms (Compose): The transformations to be applied to the image.
        vae (AutoencoderKL): The VAE model used for encoding.

    Returns:
        torch.Tensor: The latent representation of the image.
    """
    with torch.no_grad():
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            pixel_values = train_transforms(img)
        # Collate
        pixel_values = torch.stack([pixel_values])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(vae.device)
        # Encode
        latents = vae.encode(pixel_values).latent_dist.mode()
        latents *= vae.config.scaling_factor

    return latents


def compute_px_stats(
    list_images_path: List[str],
    vae: AutoencoderKL,
    train_transforms: Compose,
    latents_dim: Tuple[int, int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pixel statistics for a given VAE model and image directory.

    Args:
        list_images_path (List[str]): List of file paths to the images.
        vae (AutoencoderKL): The VAE model.
        train_transforms (Compose): The transformations applied to the training images.
        latents_dim (Tuple[int, int, int, int]): Dimensions of the latent space.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and variance of the pixel values.
    """

    latents = torch.zeros(
        (
            len(list_images_path),
            vae.config.latent_channels,
            latents_dim[-2],
            latents_dim[-1],
        ),
        device=vae.device,
    )
    for i, file in enumerate(tqdm.tqdm(list_images_path)):
        latents[i, :, :, :] = img_path_to_latents(file, train_transforms, vae)
    mean_pixels = torch.mean(latents, dim=0)
    var_pixels = torch.var(latents, dim=0)
    return mean_pixels, var_pixels


def compute_dct_px_stats(
    list_images_path: list,
    n_components: int,
    vae: AutoencoderKL,
    train_transforms: Compose,
    latents_dim: Tuple[int, int, int, int],
):
    """
    Compute the statistics of low-frequency (LF) and high-frequency (HF) components
    in the DCT (Discrete Cosine Transform) and pixel domains, respectively.

    Args:
        list_images_path (list): List of image file paths.
        n_components (int): Number of LF components to consider.
        vae (AutoencoderKL): The VAE model.
        train_transforms (Compose): Transformations applied to the input images during training.
        latents_dim (tuple): Dimensions of the latent space.

    Returns:
        mean_lf_components (torch.Tensor): Tensor of shape (n_components * latent_channels,) containing the mean of the LF components.
        mean_hf_remainder (torch.Tensor): Tensor of shape (latent_channels, resolution, resolution) containing the mean of the HF remainder.
        cov_lf_components (torch.Tensor): Tensor of shape (n_components * latent_channels, n_components * latent_channels) containing the covariance of the LF components.
        var_hf_remainder (torch.Tensor): Tensor of shape (latent_channels, resolution, resolution) containing the element-wise variance of the HF remainder.
    """

    x, y = np.meshgrid(np.arange(latents_dim[-2]), np.arange(latents_dim[-1]))
    k = np.sqrt(x**2 + y**2)

    # Get indices (DCT coordinates) of the n_components with lowest frequency -> top left quarter circle
    ind_x, ind_y = np.unravel_index(np.argsort(k, axis=None)[:n_components], k.shape)

    lf_components = torch.zeros(
        (len(list_images_path), n_components * vae.config.latent_channels),
        device=vae.device,
    )
    hf_remainders = torch.zeros(
        (
            len(list_images_path),
            vae.config.latent_channels,
            latents_dim[-2],
            latents_dim[-1],
        ),
        device=vae.device,
    )

    for i, file in enumerate(tqdm.tqdm(list_images_path)):
        latents = img_path_to_latents(file, train_transforms, vae)

        # Compute DCT
        latents = latents.cpu().numpy()
        dct = dct2(latents)

        # LF terms -> in frequency domain -> top-left quarter circle
        lf_terms = dct[0, :, ind_x, ind_y]
        # HF terms -> in pixel-domain
        dct_lf = np.zeros_like(dct)
        dct_lf[0, :, ind_x, ind_y] = lf_terms
        hf_remainder = latents[0, :, :, :] - idct2(dct_lf)[0, :, :, :]

        # Store for stats
        lf_components[i, :] = torch.from_numpy(lf_terms.flatten()).to(vae.device)
        hf_remainders[i, :, :, :] = torch.from_numpy(hf_remainder).to(vae.device)

    mean_lf_components = torch.mean(lf_components, dim=0)
    mean_hf_remainder = torch.mean(hf_remainders, dim=0)
    cov_lf_components = torch.cov(lf_components.transpose(0, 1))
    var_hf_remainder = torch.var(hf_remainders, dim=0)

    return (
        mean_lf_components,
        mean_hf_remainder,
        cov_lf_components,
        var_hf_remainder,
    )


# Main function
def main(args=None):
    if args is None:
        args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=False)

    # Load models
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=args.cache_dir, subfolder="vae"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    # Preprocessing transformations
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # List all images from the data directory
    extensions = (".png", ".jpg", ".jpeg")
    list_images_path = []
    for extension in extensions:
        list_images_path += glob.glob(f"{args.data_dir}/*{extension}")

    # Get the size of latents
    latents_dim = torch.Size(
        [1, vae.config.latent_channels, args.resolution // 8, args.resolution // 8]
    )

    # Compute stats depending on the model
    assert args.statistic_type in ["dct+pixel", "pixel"]
    if args.statistic_type == "dct+pixel":
        # Compute statistics of low-frequency (LF) and high-frequency (HF) components
        (
            mean_lf_components,
            mean_hf_remainder,
            cov_lf_components,
            var_hf_remainder,
        ) = compute_dct_px_stats(
            list_images_path, args.n_components, vae, train_transforms, latents_dim
        )

        # Save stats to output directory
        tensors = {
            "mean_lf_components": mean_lf_components,
            "mean_hf_remainder": mean_hf_remainder,
            "cov_lf_components": cov_lf_components,
            "var_hf_remainder": var_hf_remainder,
        }
        save_file(tensors, f"{args.output_dir}/stats.safetensors")

        type_file = open(f"{args.output_dir}/statistic_type.txt", "w")
        type_file.write(args.statistic_type)
        type_file.close()

    elif args.statistic_type == "pixel":
        # Compute statistics in the pixel domain (each element of the latents)
        mean_pixels, var_pixels = compute_px_stats(
            list_images_path, vae, train_transforms, latents_dim
        )

        # Save stats to output directory
        tensors = {
            "mean_pixels": mean_pixels,
            "var_pixels": var_pixels,
        }
        save_file(tensors, f"{args.output_dir}/stats.safetensors")

        type_file = open(f"{args.output_dir}/statistic_type.txt", "w")
        type_file.write(args.statistic_type)
        type_file.close()


if __name__ == "__main__":
    main()
