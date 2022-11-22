import os
from io import BytesIO
import base64

import torch
from diffusers import PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from PIL import Image, ImageOps
from cog import BasePredictor, Input, Path

import pipelines


torch.set_grad_enabled(False)


def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__


# patch_conv(padding_mode='circular')

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        model_name = "podium2-bg_style"
        self.pipe = pipelines.StableDiffusionPipeline.from_pretrained(
            model_name,
        ).to("cuda")
        self.pipe.disable_nsfw_filter()
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            model_name,
            subfolder="scheduler",
            solver_order=2,
            predict_epsilon=True,
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,
        )

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
        output_type: str = Input(
            description="Type of ponzu output requested", choices=["icon", "bg", "default"], default="default"
        )
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
            if output_type == "icon":
                self.pipe.scheduler = PNDMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
                )

        if mask:
            mask = Image.open(
                BytesIO(base64.b64decode(mask))).convert("RGB")
            mask = ImageOps.invert(mask)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt if prompt is not None else None,
            negative_prompt=negative_prompt if negative_prompt is not None else None,
            num_images_per_prompt=num_outputs,
            height=height,
            width=width,
            init_image=init_image,
            mask_image=mask,
            strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        torch.cuda.empty_cache()
        return dict(seed=seed, data=output_paths)
