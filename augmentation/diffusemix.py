"""
DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models
Reference: Islam et al., CVPR 2024 - "DiffuseMix: Label-Preserving Data Augmentation
           with Diffusion Models"

Key idea: Generate a fractal/texture image via a diffusion process, then
linearly blend it with the original training image while keeping the original
label. This forces the network to rely on structural features rather than
superficial texture cues.

When the optional `diffusers` library is present the implementation drives
an actual lightweight diffusion model (runwayml/stable-diffusion-v1-5 at
very few denoising steps for speed).  When diffusers is not installed it
falls back to a high-quality procedural alternative that reproduces the
mixing behaviour without the generative model.
"""

import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _DIFFUSERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Procedural fractal / Perlin-noise texture generator
# (used when diffusers is not available or for fast CPU fallback)
# ---------------------------------------------------------------------------

def _smooth_noise_2d(width: int, height: int, scale: float) -> np.ndarray:
    """Return a smooth noise map in [0, 1] at the given spatial scale."""
    small_w = max(1, int(width / scale))
    small_h = max(1, int(height / scale))
    rng = np.random.default_rng()
    noise = rng.random((small_h, small_w)).astype(np.float32)
    img = Image.fromarray((noise * 255).astype(np.uint8), mode="L")
    img = img.resize((width, height), Image.BICUBIC)
    return np.array(img).astype(np.float32) / 255.0


def _fractal_texture(width: int, height: int,
                     octaves: int = 6,
                     persistence: float = 0.5,
                     lacunarity: float = 2.0) -> np.ndarray:
    """
    Multi-octave fractal (Perlin-style) noise texture.
    Returns an RGB array of shape (H, W, 3) in [0, 1].
    """
    channels = []
    for _ in range(3):
        accumulated = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            scale = max(width, height) / frequency
            accumulated += amplitude * _smooth_noise_2d(width, height, scale)
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        channels.append(accumulated / max_value)
    texture = np.stack(channels, axis=-1)          # (H, W, 3)
    # normalise to [0, 1]
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    return texture


def _style_prompts():
    """Random style descriptors used when prompting the diffusion model."""
    return random.choice([
        "sunset lighting, warm tones",
        "rainbow colors, vibrant",
        "rainy window, blurred background",
        "neon glow, cyberpunk style",
        "watercolor painting style",
        "pencil sketch style",
        "oil painting texture",
        "soft bokeh, dreamy atmosphere",
        "black and white high contrast",
        "vintage film grain effect",
    ])


# ---------------------------------------------------------------------------
# DiffuseMix class
# ---------------------------------------------------------------------------

class DiffuseMix:
    """
    DiffuseMix augmentation transform.

    Parameters
    ----------
    alpha : float
        Blending coefficient λ ∈ [0, 1].  The augmented image is
        ``λ * x_orig + (1-λ) * x_generated``.  Original label is preserved.
    strength : float
        Denoising strength for the img2img pipeline [0, 1].  Ignored when
        diffusers is not available.
    use_diffusion : bool
        Force-enable or force-disable the diffusion backend regardless of
        library availability.
    device : str
        Device for the diffusion pipeline ("cuda" / "cpu").
    """

    def __init__(
        self,
        alpha: float = 0.5,
        strength: float = 0.6,
        use_diffusion: bool = True,
        device: str = "cpu",
    ):
        self.alpha = alpha
        self.strength = strength
        self.device = device
        self.pipeline = None

        if use_diffusion and _DIFFUSERS_AVAILABLE:
            try:
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                    safety_checker=None,
                ).to(device)
                self.pipeline.set_progress_bar_config(disable=True)
            except Exception:
                self.pipeline = None   # fall back silently

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply DiffuseMix to a PIL image.

        Returns the blended PIL image.
        """
        if self.pipeline is not None:
            generated = self._diffusion_augment(image)
        else:
            generated = self._procedural_augment(image)

        # Blend: λ * original + (1-λ) * generated
        blended = Image.blend(image.convert("RGB"),
                              generated.convert("RGB"),
                              1.0 - self.alpha)
        return blended

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _diffusion_augment(self, image: Image.Image) -> Image.Image:
        """Drive the img2img pipeline with a random style prompt."""
        prompt = _style_prompts()
        result = self.pipeline(
            prompt=prompt,
            image=image.convert("RGB"),
            strength=self.strength,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
        return result

    def _procedural_augment(self, image: Image.Image) -> Image.Image:
        """
        CPU fallback: produce a texture image that mimics the variety a
        diffusion model would generate.  Uses multi-octave fractal noise
        tinted with a random colour palette.
        """
        w, h = image.size
        texture = _fractal_texture(w, h)

        # Random colour tinting  — mimic style diversity
        tint = np.random.uniform(0.4, 1.0, size=(1, 1, 3)).astype(np.float32)
        tinted = (texture * tint * 255).clip(0, 255).astype(np.uint8)
        generated = Image.fromarray(tinted, mode="RGB")

        # Optional mild blur (diffusion outputs are typically smooth)
        if random.random() < 0.5:
            generated = generated.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))
            )
        return generated


# ---------------------------------------------------------------------------
# Convenience function for use inside a DataLoader collate_fn
# ---------------------------------------------------------------------------

def apply_diffusemix_batch(
    images: torch.Tensor,
    diffusemix: DiffuseMix,
    prob: float = 0.4,
) -> torch.Tensor:
    """
    Apply DiffuseMix to a batch tensor of shape (B, C, H, W) in [0, 1].

    Each image in the batch is independently augmented with probability
    `prob`.  Returns a new tensor of the same shape.
    """
    to_pil = TF.to_pil_image
    result = images.clone()
    for i in range(images.size(0)):
        if random.random() < prob:
            pil = to_pil(images[i].cpu())
            augmented = diffusemix(pil)
            result[i] = TF.to_tensor(augmented)
    return result
