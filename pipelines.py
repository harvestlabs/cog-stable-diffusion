from typing import List, Optional, Union

import PIL
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline as StableDiffusionText2ImgPipeline,
    UNet2DConditionModel
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class DummySafetyChecker:
    @staticmethod
    def __call__(images, clip_input):
        return images, False


class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()

        text2img = StableDiffusionText2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        img2img = StableDiffusionImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=safety_checker,
        )

        self.register_modules(
            text2img=text2img,
            img2img=img2img,
            inpaint=inpaint,
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def disable_nsfw_filter(self):
        self.safety_checker = DummySafetyChecker()
        self.text2img.safety_checker = self.safety_checker
        self.img2img.safety_checker = self.safety_checker
        self.inpaint.safety_checker = self.safety_checker

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
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
            result = self.text2img(
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
            if mask_image is None:
                result = self.img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    generator=generator,
                    output_type=output_type,
                    **kwargs,
                )
            else:
                result = self.inpaint(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    mask_image=mask_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    generator=generator,
                    output_type=output_type,
                    **kwargs,
                )

        return result
