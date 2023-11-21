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

    channel_means: List[float] = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
    channel_stds: List[float] = [0.9654121, 1.0440036, 0.76147926, 0.77022034]
    self.normalize = Normalize(channel_means, channel_stds)
  
  def get_eps(self, sample: FloatTensor, timestep: LongTensor, latents: FloatTensor) -> FloatTensor:
    return super().__init__(sample, timestep)