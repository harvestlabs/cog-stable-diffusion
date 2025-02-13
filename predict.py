from typing import List, Optional
import os
from io import BytesIO
import base64

import torch
import controlnet_hinter
from diffusers import PNDMScheduler
from PIL import Image, ImageOps
from cog import BasePredictor, Input, Path
import numpy as np

import pipelines


torch.set_grad_enabled(False)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = pipelines.HarvestLabsPipelines()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        negative_prompt: str = Input(default=None),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        init_image: str = Input(
            description="Inital image to generate variations of. Will be resized to the specified width and height",
            default=None,
        ),
        mask: str = Input(
            description="Black and white image to use as mask for inpainting over init_image. Black pixels are inpainted and white pixels are preserved. Experimental feature, tends to work better with prompt strength of 0.5-0.7",
            default=None,
        ),
        depth_image: str = Input(
            description="Depth mask of the init_image",
            default=None,
        ),
        canny_image: str = Input(
            description="Normal of the init_image",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 2, 3, 4, 5, 10], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        model: str = Input(default="base", choices=[
                           "base", "depth", "upscale", "skybox"]),
        control_conditioning: str = Input(default=None),
    ) -> dict:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        if init_image:
            init_image = Image.open(
                BytesIO(base64.b64decode(init_image))).convert("RGB")

        if depth_image:
            depth_image = Image.open(
                BytesIO(base64.b64decode(depth_image))).convert("RGB")
        if canny_image:
            canny_image = Image.open(
                BytesIO(base64.b64decode(canny_image))).convert("RGB")
            canny_image = controlnet_hinter.hint_fake_scribble(canny_image)
        if mask:
            mask = Image.open(
                BytesIO(base64.b64decode(mask))).convert("L")
        if control_conditioning:
            control_conditioning = [float(i)
                                    for i in control_conditioning.split(",")]

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt if prompt is not None else None,
            negative_prompt=negative_prompt if negative_prompt is not None else None,
            num_images_per_prompt=num_outputs,
            height=height,
            width=width,
            init_image=init_image,
            depth_image=depth_image,
            canny_image=canny_image,
            mask_image=mask,
            strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            model=model,
            conditioning=control_conditioning[0] if control_conditioning else None,
            conditioning2=control_conditioning[1] if control_conditioning else None,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        torch.cuda.empty_cache()
        return dict(seed=seed, data=output_paths)
