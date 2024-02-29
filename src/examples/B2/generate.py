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

import os
import torch
from diffusers import StableDiffusionPipeline
from signal_leak import sample_from_stats

folder = "examples/B2/imgs"
path_stats = "examples/A2"

os.makedirs(folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
).to(device)
num_inference_steps = 50

# Get the timestep T of the first reverse diffusion iteration
pipeline.scheduler.set_timesteps(num_inference_steps, device="cuda")
first_inference_timestep = pipeline.scheduler.timesteps[0].item()

# Get the values of sqrt(alpha_prod_T)
sqrt_alpha_prod = pipeline.scheduler.alphas_cumprod[first_inference_timestep] ** 0.5
sqrt_one_minus_alpha_prod = (1 - pipeline.scheduler.alphas_cumprod[first_inference_timestep]) ** 0.5

# Dimensions of the latent space, with batch_size=1
shape_latents = [
    1,
    pipeline.unet.config.in_channels,
    pipeline.unet.config.sample_size,
    pipeline.unet.config.sample_size,
]

# Utility function fo visualize initial latents / signal leak
def latents_to_pil(pipeline, latents, generator):
    decoded = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False,
        generator=generator,
    )[0]
    image = pipeline.image_processor.postprocess(
        decoded,
        output_type="pil",
        do_denormalize=[True],
    )[0]
    return image

# Random number generator
generator = torch.Generator(device=device)
generator = generator.manual_seed(12345)
        
with torch.no_grad():
    for n in range(5):

        # Generate the initial latents
        initial_latents = torch.randn(
            shape_latents, generator=generator, device=device, dtype=torch.float32
        )
        latents_to_pil(pipeline, initial_latents, generator).save(f"{folder}/latents{n}.png")
        

        # Generate an image WITHOUT signal-leak in the initial latents
        image = pipeline(
            prompt="An astronaut riding a horse, in the style of line art, pastel colors.",
            num_inference_steps=num_inference_steps,
            latents=initial_latents,
        ).images[0]
        image.save(f"{folder}/original{n}.png")

        # Generate a signal leak from computed statistics
        signal_leak = sample_from_stats(
            path=path_stats,
            dims=shape_latents,
            generator_pt=generator,
            generator_np=None,
            device=device
        )
        latents_to_pil(pipeline, signal_leak, generator).save(f"{folder}/signal_leak{n}.png")
        
        # Add a signal leak in the initial latents
        initial_latents = sqrt_alpha_prod * signal_leak + sqrt_one_minus_alpha_prod * initial_latents

        # Generate an image WITH signal-leak in the initial latents
        image_with_signalleak = pipeline(
            prompt="An astronaut riding a horse, in the style of line art, pastel colors.",
            num_inference_steps=num_inference_steps,
            latents=initial_latents,
        ).images[0]
        image_with_signalleak.save(f"{folder}/ours{n}.png")