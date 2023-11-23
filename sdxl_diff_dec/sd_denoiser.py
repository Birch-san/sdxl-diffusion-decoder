from diffusers import UNet2DConditionModel
from torch import FloatTensor, LongTensor, cat
import torch
from typing import List

from .denoiser import DiffusersDenoiser
from .schedule import betas_for_alpha_bar, alpha_bar, get_alphas
from .normalize import Normalize
from k_diffusion.sampling import append_zero

class SDDecoder(DiffusersDenoiser):
  normalize: Normalize
  timesteps: LongTensor
  rounded_timesteps: LongTensor
  rounded_sigmas: FloatTensor
  rounded_log_sigmas: FloatTensor
  def __init__(self, unet: UNet2DConditionModel, dtype: torch.dtype = None):
    total_timesteps=1024
    betas = betas_for_alpha_bar(total_timesteps, alpha_bar, device=unet.device)
    alphas: FloatTensor = get_alphas(betas)
    alphas_cumprod: FloatTensor = alphas.cumprod(dim=0)
    super().__init__(unet, alphas_cumprod, dtype=dtype)

    # channel_means: List[float] = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
    # channel_stds: List[float] = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

    # divided by 0.18215, so we don't have to do that bizarre "standardize the wonky way, then scale-and-shift from there"
    channel_means: List[float] = [2.1335418224334717, 0.12369272112846375, 0.4052227735519409, -0.09404008090496063],
    channel_stds: List[float] = [5.300093650817871, 5.731559753417969, 4.180506229400635, 4.228494644165039],
    self.normalize = Normalize(channel_means, channel_stds).to(unet.device)

    n_distilled_steps=64
    space: int = total_timesteps//n_distilled_steps
    self.timesteps = torch.arange(0, total_timesteps, device=unet.device)
    self.rounded_timesteps = ((self.timesteps//space)+1).clamp_max(n_distilled_steps-1)*space
    self.rounded_sigmas = self.t_to_sigma(self.rounded_timesteps)
    self.rounded_log_sigmas = self.rounded_sigmas.log()
  
  def get_v(self, sample: FloatTensor, timestep: LongTensor, latents: FloatTensor) -> FloatTensor:
    noise_and_latents: FloatTensor = cat([sample, latents], dim=1)
    out: FloatTensor = super().get_v(noise_and_latents, timestep)
    out = out[:, :3, :, :]
    return out

  def get_sigmas_rounded(self, include_sigma_min=True, n=None) -> FloatTensor:
    if n is None:
      return append_zero(self.rounded_sigmas.flip(0))
    t_max: int = len(self.rounded_sigmas) - 1
    t: FloatTensor = torch.linspace(t_max, 0, n, device=self.rounded_sigmas.device)
    if not include_sigma_min:
      t = t[:-1]
    return append_zero(self.t_to_sigma_rounded(t))
  
  def t_to_sigma_rounded(self, t: LongTensor) -> FloatTensor:
    t: FloatTensor = t.float()
    low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
    log_sigma = (1 - w) * self.rounded_log_sigmas[low_idx] + w * self.rounded_log_sigmas[high_idx]
    return log_sigma.exp()