# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

'''
code review
authors
- Wei Jiang, Suanfamama, wei@suanfamama.com
- Mama Xiao, Suanfamama, mama.xiao@suanfamama.com

This Python program is designed to generate new images using a pre-trained Diffusion Image Text (DiT) model. Here's a breakdown of its functionality:

1. Setup and Initialization

* Imports: The program imports necessary libraries like torch, torchvision, diffusion, diffusers, download, and models.
* Device Selection: It determines whether to use a GPU or CPU for computation.
* Model Loading:
    * It defines the DiT model architecture based on the chosen model type (args.model).
    * It either loads a pre-trained DiT-XL/2 model (if args.ckpt is not provided) or loads a custom checkpoint from a previous training session.
    * It loads a pre-trained VAE (Variational Autoencoder) for image reconstruction.
* Class Labels: It defines a list of class labels (e.g., image categories) to condition the image generation process.
* Noise Generation: It creates random noise tensors (z) to be used as input to the diffusion process.
* Classifier-Free Guidance: It sets up classifier-free guidance by duplicating the noise and labels, allowing the model to generate images that are more aligned with the specified class labels.

2. Image Sampling

* Diffusion Sampling: The program uses the diffusion.p_sample_loop function to generate images by iteratively denoising the noise tensors (z) using the DiT model.
* VAE Reconstruction: The generated latent representations are decoded using the VAE to produce actual images.

3. Image Saving and Display:

* Image Saving: The generated images are saved to a file named "sample.png".
* Image Display: The images are displayed in a grid format.
In essence, this program takes a set of class labels as input and generates new images that correspond to those labels using a pre-trained DiT model and a diffusion process.

Key Concepts:

* Diffusion Models: Diffusion models are a type of generative model that learn to reverse a noisy process to generate new data.
* DiT (Diffusion Image Text): DiT models are diffusion models specifically designed for image generation, often conditioned on text prompts.
* Classifier-Free Guidance: A technique used to improve the quality and controllability of generated images by providing guidance from a classifier (in this case, the class labels).
* VAE (Variational Autoencoder): A type of neural network that learns to compress and reconstruct data, often used for image generation and encoding.

This program provides a basic example of how to use a pre-trained DiT model for image generation. We can modify the code to experiment with different model architectures, class labels, and hyperparameters to explore the capabilities of diffusion models.

## Improvements

1. Code Structure and Readability

Function Decomposition: The main function is quite long and could benefit from being broken down into smaller, more focused functions. This would improve code organization and make it easier to understand and maintain. For example, we could create separate functions for:
* load_model_and_diffusion()
* setup_sampling_parameters()
* generate_images()
* save_and_display_images()

Docstrings: Add comprehensive docstrings to your functions and classes. This helps explain what each part of the code does and makes it easier for others (and your future self) to understand.

Variable Naming: Use descriptive variable names that clearly indicate their purpose. For example, instead of z, use noise_tensors.

2. Error Handling and Robustness

Input Validation: Add checks to ensure that the user-provided arguments are valid. For example, you could check that the image_size is one of the allowed values (256 or 512).

Exception Handling: Implement try-except blocks to handle potential errors during model loading, image generation, or saving. This can help prevent the program from crashing unexpectedly.

Logging: Add logging statements to record important events during the program execution, such as model loading, sampling parameters, and image generation. This can help with debugging and troubleshooting.

3. Performance Optimization

with torch.no_grad():: Wrap the image generation loop with with torch.no_grad(): to disable gradient calculations, as they are not needed during sampling. This can improve performance.

torch.cuda.empty_cache(): Call torch.cuda.empty_cache() after the image generation to free up GPU memory.

4. Additional Considerations

Hyperparameter Tuning: Provide more flexibility for users to adjust hyperparameters like cfg_scale and num_sampling_steps through command-line arguments.

Image Quality: Experiment with different VAE models (e.g., stabilityai/sd-vae-ft-ema) to see if they produce better image quality.

User Interface: Consider creating a more user-friendly interface (e.g., a GUI or a web application) to make it easier for users to generate images without needing to modify the code.
'''