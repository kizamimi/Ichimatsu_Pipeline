import copy
import torch
import PIL.Image
import numpy as np
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Callable, List, Optional, Union
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor

def _encode_prompt(
    self,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    **kwargs,
):
    prompt_embeds_tuple = self.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        **kwargs,
    )

    # concatenate for backwards comp
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    else:
        prompt_embeds = prompt_embeds_tuple[0]

    return prompt_embeds
StableDiffusionImg2ImgPipeline._encode_prompt = _encode_prompt

def decode_latents(self, latents):
    float_type = latents.dtype
    latents = 1 / self.vae.config.scaling_factor * latents
    self.vae = self.vae.to(torch.float32)
    latents = latents.to(torch.float32)
    image = self.vae.decode(latents, return_dict=False)[0]
    self.vae = self.vae.to(float_type)
    image = image.to(float_type)
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
StableDiffusionImg2ImgPipeline.decode_latents = decode_latents

def prepare_latents(self, image, batch_size, num_images_per_prompt, dtype, device, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if isinstance(generator, list):
        init_latents = [
            self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
        ]
        init_latents = torch.cat(init_latents, dim=0)
    else:
        init_latents = self.vae.encode(image).latent_dist.sample(generator)

    init_latents = self.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        # deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return init_latents, noise
StableDiffusionImg2ImgPipeline.prepare_latents = prepare_latents

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]

def denoiser_base(self, latents, guidance_scale, do_classifier_free_guidance, \
                  prompt_embeds, timesteps, num_inference_steps, num_warmup_steps):
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
    return latents
StableDiffusionImg2ImgPipeline.denoiser_base = denoiser_base

def denoiser(self, latents, scale_noise, ichimatsu, \
               do_classifier_free_guidance, \
               prompt_embeds, guidance_scale, t):
    self.scheduler = copy.deepcopy(self.scheduler_step)

    noised_latents = latents * ichimatsu + ( latents + scale_noise ) * ( 1 - ichimatsu )

    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([noised_latents] * 2) if do_classifier_free_guidance else noised_latents
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond
        noise_pred_text = noise_pred_text
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred
StableDiffusionImg2ImgPipeline.denoiser = denoiser

def denoiser_ichimatsu(self, image, random_noise, guidance_scale, do_classifier_free_guidance, \
                  prompt_embeds, timesteps, num_inference_steps, num_warmup_steps):
    with self.progress_bar(total=num_inference_steps) as progress_bar:

        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.scheduler_copy = copy.deepcopy(self.scheduler)

        height = image.shape[2]
        width = image.shape[3]

        ichimatsu = []
        for b in range(4):
            pix = []
            for i in range(height):
                one_line = []
                for j in range(width):
                    if j%2+i%2*2 == b:
                        one_line.append(0)
                    else:
                        one_line.append(1)
                pix.append(one_line)
            pix = torch.Tensor(pix).to(torch.float16).to(device)
            ichimatsu.append(pix)

        self.scheduler_step = copy.deepcopy(self.scheduler_copy)
        self.scheduler = copy.deepcopy(self.scheduler_copy)

        self.scheduler = copy.deepcopy(self.scheduler_step)
        latents = self.scheduler.add_noise(image, torch.zeros_like(random_noise), torch.LongTensor([timesteps[0]]).to(torch.long).to(device))

        self.scheduler = copy.deepcopy(self.scheduler_step)
        scaled_noise = self.scheduler.add_noise(torch.zeros_like(random_noise), random_noise, torch.LongTensor([timesteps[0]]).to(torch.long).to(device))

        for i, t in enumerate(timesteps):
            
            if i >= 20:
                self.scheduler = copy.deepcopy(self.scheduler_step)

                noised_latents = latents + scaled_noise

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([noised_latents] * 2) if do_classifier_free_guidance else noised_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond = noise_pred_uncond
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                
                noise_pred = self.denoiser(latents, scaled_noise, ichimatsu[0], \
                    do_classifier_free_guidance, prompt_embeds, guidance_scale, t)
                
                noise_pred2 = self.denoiser(latents, scaled_noise, ichimatsu[1], \
                    do_classifier_free_guidance, prompt_embeds, guidance_scale, t)
                
                noise_pred3 = self.denoiser(latents, scaled_noise, ichimatsu[2], \
                    do_classifier_free_guidance, prompt_embeds, guidance_scale, t)
                
                noise_pred4 = self.denoiser(latents, scaled_noise, ichimatsu[3], \
                    do_classifier_free_guidance, prompt_embeds, guidance_scale, t)

                noise_pred = noise_pred * ( 1 - ichimatsu[0] ) + noise_pred2 * ( 1 - ichimatsu[1] ) + noise_pred3 * ( 1 - ichimatsu[2] )  + noise_pred4 * ( 1 - ichimatsu[3] )

            self.scheduler = copy.deepcopy(self.scheduler_step)
            latents = self.scheduler.step(noise_pred/2, t, latents, **self.extra_step_kwargs).prev_sample

            self.scheduler = copy.deepcopy(self.scheduler_step)
            scaled_noise = self.scheduler.step(noise_pred/2, t, scaled_noise, **self.extra_step_kwargs).prev_sample

            self.scheduler_step = copy.deepcopy(self.scheduler)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    return latents + scaled_noise
StableDiffusionImg2ImgPipeline.denoiser_ichimatsu = denoiser_ichimatsu

def prompt_filter_main(self, image, random_noise, prompt_embeds_list, \
                       guidance_scale, do_classifier_free_guidance, timesteps,\
                       num_inference_steps, num_warmup_steps, use_ichimatsu_pipeline):
    if use_ichimatsu_pipeline:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        latents = self.denoiser_ichimatsu(image, random_noise, guidance_scale, \
                                        do_classifier_free_guidance, prompt_embeds_list[0], \
                                        timesteps, num_inference_steps, num_warmup_steps)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        latents = self.scheduler.add_noise(image, random_noise, torch.LongTensor([timesteps[0]]).to(device))
        latents = self.denoiser_base(latents, guidance_scale, do_classifier_free_guidance, \
                                        prompt_embeds_list[0], timesteps, num_inference_steps, \
                                        num_warmup_steps)
    return latents
StableDiffusionImg2ImgPipeline.prompt_filter_main = prompt_filter_main

@torch.no_grad()
def prompt_filter_call(
    self,
    prompt: Union[str, List[str]] = None,
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    strength: float = 0.8,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: Optional[float] = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    use_ichimatsu_pipeline: bool = False,
    **kwargs,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        self._execution_device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    ).to(device).to(torch.float16)
    root_prompt_embeds = self._encode_prompt(
        "",
        self._execution_device,
        num_images_per_prompt,
        do_classifier_free_guidance = False
    ).to(device).to(torch.float16)
    prompt_embeds_list = [prompt_embeds, root_prompt_embeds]

    # 4. Preprocess image
    image = self.image_processor.preprocess(image).to(device).to(torch.float16)

    # 5. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)

    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, self._execution_device)

    # 6. Prepare latent variables
    init_latent, random_noise = self.prepare_latents(
        image, batch_size, num_images_per_prompt, prompt_embeds.dtype, self._execution_device, generator
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    latents = self.prompt_filter_main(init_latent, random_noise,\
                                    prompt_embeds_list, guidance_scale, \
                                    do_classifier_free_guidance, timesteps, \
                                    num_inference_steps, num_warmup_steps, \
                                    use_ichimatsu_pipeline)

    # 9. Post-processing
    image = self.decode_latents(latents)

    # 10. Run safety checker
    image, has_nsfw_concept = self.run_safety_checker(image, self._execution_device, prompt_embeds.dtype)

    # 11. Convert to PIL
    if output_type == "pil":
        image = self.numpy_to_pil(image)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
StableDiffusionImg2ImgPipeline.__call__ = prompt_filter_call

from safetensors.torch import load_file
def load_safetensors_lora(pipeline, checkpoint_path, LORA_PREFIX_UNET="lora_unet", LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=0.75):
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
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
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

def get_pipe(pretrained_model_name_or_path, lora_checkpoint_path="", vae_path="", torch_dtype=torch.float16, alpha=0.75):

    # Stable Diffusion
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(pretrained_model_name_or_path, torch_dtype=torch_dtype)
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)

    # Lora
    if not lora_checkpoint_path == "":
        if not ".safetensor" in lora_checkpoint_path:
            pipe.unet.load_attn_procs(lora_checkpoint_path)
        else:
            pipe = load_safetensors_lora(pipe, lora_checkpoint_path, alpha=alpha)

    # VAE
    if not vae_path == "":
        pipe.vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch_dtype)

    pipe = pipe.to("cuda")
    return pipe
