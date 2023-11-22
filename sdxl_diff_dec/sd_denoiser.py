from diffusers import UNet2DConditionModel
from torch import FloatTensor, LongTensor
import torch
from typing import List

from .denoiser import DiffusersDenoiser
from .schedule import betas_for_alpha_bar, alpha_bar, get_alphas
from .normalize import Normalize

class SDDecoder(DiffusersDenoiser):
  normalize: Normalize
  def __init__(self, unet: UNet2DConditionModel, dtype: torch.dtype = None):
    betas = betas_for_alpha_bar(1024, alpha_bar, device=unet.device)
    alphas: FloatTensor = get_alphas(betas)
    alphas_cumprod: FloatTensor = alphas.cumprod(dim=0)
    super().__init__(unet, alphas_cumprod, dtype=dtype)

    # channel_means: List[float] = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
    # channel_stds: List[float] = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

    # divided by 0.18215, so we don't have to do that bizarre "standardize the wonky way, then scale-and-shift from there"
    channel_means: List[float] = [2.1335418224334717, 0.12369272112846375, 0.4052227735519409, -0.09404008090496063],
    channel_stds: List[float] = [5.300093650817871, 5.731559753417969, 4.180506229400635, 4.228494644165039],
    self.normalize = Normalize(channel_means, channel_stds)

    total_timesteps=1024
    n_distilled_steps=64
    space: int = total_timesteps//n_distilled_steps
    timesteps: LongTensor = torch.arange(0, 1024, device=unet.device)
    rounded_timesteps: LongTensor = ((timesteps//space)+1).clamp_max(n_distilled_steps-1)*space
  
  def get_eps(self, sample: FloatTensor, timestep: LongTensor, latents: FloatTensor) -> FloatTensor:
    return super().__init__(sample, timestep)