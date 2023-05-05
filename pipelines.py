from typing import List, Optional, Union

import numpy as np
import os
import PIL
import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionUpscalePipeline,
)

os.makedirs("diffusers-cache", exist_ok=True)


def replacementConv2DConvForward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    working = F.pad(input, self.paddingX, mode='circular')
    working = F.pad(working, self.paddingY, mode='constant')
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


def patch_conv():
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.padding_modeX = 'circular'
        self.padding_modeY = 'constant'
        self.paddingX = (
            self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
        self.paddingY = (
            0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
        self.paddingStartStep = 15
        self.paddingStopStep = -1
        self._conv_forward = replacementConv2DConvForward.__get__(
            self, torch.nn.Conv2d)
    cls.__init__ = __init__


def restore_conv():
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.padding_modeX = 'constant'
        self.padding_modeY = 'constant'
        self._conv_forward = torch.nn.Conv2d._conv_forward.__get__(
            self, torch.nn.Conv2d)

    cls.__init__ = __init__


def load_lora_weights(pipeline, checkpoint_path):
    from safetensors.torch import load_file
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(
                LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(
                LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(
                3).squeeze(2).to(torch.float16)
            weight_down = state_dict[pair_keys[1]].squeeze(
                3).squeeze(2).to(torch.float16)
            curr_layer.weight.data += alpha * \
                torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float16)
            weight_down = state_dict[pair_keys[1]].to(torch.float16)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


class HarvestLabsPipelines:
    def __init__(
        self,
    ):
        super().__init__()

        # Do tile the X
        patch_conv()
        base_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", cache_dir="diffusers-cache", torch_dtype=torch.float16)
        base_controlnet2 = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", cache_dir="diffusers-cache", torch_dtype=torch.float16)
        self.base = StableDiffusionControlNetPipeline.from_pretrained(
            "darkstorm2150/Protogen_x3.4_Official_Release",
            cache_dir="diffusers-cache",
            safety_checker=None,
            controlnet=base_controlnet,
            controlnet2=base_controlnet2,
            torch_dtype=torch.float16,
        )
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config)
        self.base.enable_xformers_memory_efficient_attention()
        self.base.to("cuda")
        # self.base = load_lora_weights(self.base, "./latent360.safetensors")

        # Don't tile the X
        restore_conv()
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-depth-diffusers", cache_dir="diffusers-cache", torch_dtype=torch.float16)
        controlnet2 = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-scribble-diffusers", cache_dir="diffusers-cache", torch_dtype=torch.float16)
        self.control = StableDiffusionControlNetPipeline.from_pretrained(
            "new_style",
            safety_checker=None,
            controlnet=controlnet,
            controlnet2=controlnet2,
            torch_dtype=torch.float16
        )
        self.control.scheduler = DPMSolverMultistepScheduler.from_config(
            self.control.scheduler.config)
        self.control.enable_xformers_memory_efficient_attention()
        self.control.to("cuda")

        self.upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            cache_dir="diffusers-cache",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        self.upscaler.scheduler = DPMSolverMultistepScheduler.from_config(
            self.upscaler.scheduler.config)
        self.upscaler.vae.enable_tiling()
        self.upscaler.enable_xformers_memory_efficient_attention()
        self.upscaler.to("cuda")

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        depth_image: Optional[PIL.Image.Image] = None,
        canny_image: Optional[Union[torch.FloatTensor,
                                    PIL.Image.Image]] = None,
        strength: float = 0.8,
        mask_image: Optional[Union[torch.FloatTensor,
                                   PIL.Image.Image]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        model: Optional[str] = "base",
        conditioning: Optional[float] = 0.5,
        conditioning2: Optional[float] = 0.5,
        **kwargs,
    ):

        if model == "base" or model == "skybox":
            result = self.base(
                prompt=prompt,
                control_image=depth_image,
                control_image2=canny_image,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                latents=latents,
                output_type=output_type,
                conditioning=conditioning,
                conditioning2=conditioning2,
                **kwargs,
            )
        elif model == "depth":
            result = self.control(
                prompt=prompt,
                control_image=depth_image,
                control_image2=canny_image,
                init_image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                output_type=output_type,
                conditioning=conditioning,
                conditioning2=conditioning2,
                **kwargs,
            )
        elif model == "upscale":
            result = self.upscaler(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        return result
