from typing import List, Optional, Union

import os
import PIL
import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionDepth2ImgPipeline,
)

os.makedirs("cache", exist_ok=True)


class HarvestLabsPipelines:
    def __init__(
        self,
    ):
        super().__init__()

        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            cache_dir="cache",
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16"
        )
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config)
        self.base = self.base.to("cuda")

        self.depth = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            cache_dir="cache",
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda")

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        depth_image: Optional[Union[torch.FloatTensor,
                                    PIL.Image.Image]] = None,
        strength: float = 0.8,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):

        if init_image is None:
            # txt2img
            result = self.base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                latents=latents,
                output_type=output_type,
                **kwargs,
            )
        else:
            # depth
            result = self.depth(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                depth_map=depth_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                output_type=output_type,
                **kwargs,
            )

        return result
