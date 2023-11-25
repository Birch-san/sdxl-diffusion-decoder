from k_diffusion.external import DiscreteVDDPMDenoiser
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from torch import FloatTensor, LongTensor, cat
import torch
from typing import Optional

from k_diffusion.sampling import append_zero

class SDDecoder(DiscreteVDDPMDenoiser):
  inner_model: UNet2DModel
  timesteps: LongTensor
  sampling_dtype: torch.dtype
  def __init__(
    self,
    unet: UNet2DModel,
    alphas_cumprod: FloatTensor,
    quantize=True,
    dtype: torch.dtype = None,
  ):
    super().__init__(unet, alphas_cumprod, quantize=quantize)
    self.sigma_data = .5
    self.sampling_dtype = unet.dtype if dtype is None else dtype
  
  def get_v(self, noised_rgb: FloatTensor, timestep: LongTensor, latents: FloatTensor) -> FloatTensor:
    noise_and_latents: FloatTensor = cat([noised_rgb, latents], dim=1).to(self.inner_model.dtype)
    out: UNet2DOutput = self.inner_model.forward(
      noise_and_latents,
      timestep,
    )
    sample: FloatTensor = out.sample
    # retain just the RGB output channels
    sample = sample[:, :3, :, :]
    return sample.to(self.sampling_dtype).neg()

  def forward(self, input: FloatTensor, sigma: FloatTensor, **kwargs) -> FloatTensor:
    nominal: FloatTensor = super().forward(input, sigma, **kwargs)
    # OpenAI applies static thresholding after scaling
    return nominal.clamp(-1, 1)

class SDDecoderDistilled(SDDecoder):
  rounded_timesteps: LongTensor
  rounded_sigmas: FloatTensor
  rounded_log_sigmas: FloatTensor
  def __init__(
    self,
    unet: UNet2DModel,
    alphas_cumprod: FloatTensor,
    total_timesteps=1024,
    n_distilled_steps=64,
    quantize=True,
    dtype: torch.dtype = None,
  ):
    super().__init__(unet, alphas_cumprod, quantize=quantize, dtype=dtype)

    space: int = total_timesteps//n_distilled_steps
    self.timesteps = torch.arange(0, total_timesteps, device=unet.device)
    self.rounded_timesteps = ((self.timesteps//space)+1).clamp_max(n_distilled_steps-1)*space
    self.rounded_sigmas = self.t_to_sigma(self.rounded_timesteps)
    self.rounded_log_sigmas = self.rounded_sigmas.log()

  def get_sigmas_rounded(self, include_sigma_min=True, n: Optional[int] = None) -> FloatTensor:
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