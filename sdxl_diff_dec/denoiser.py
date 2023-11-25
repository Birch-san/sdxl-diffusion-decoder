from k_diffusion.external import DiscreteVDDPMDenoiser
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from torch import FloatTensor, LongTensor
import torch

class DiffusersDenoiser(DiscreteVDDPMDenoiser):
  inner_model: UNet2DModel
  sigmas: FloatTensor
  log_sigmas: FloatTensor
  alphas_cumprod: FloatTensor
  def __init__(self, unet: UNet2DModel, alphas_cumprod: FloatTensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    self.alphas_cumprod = alphas_cumprod
    super().__init__(unet, alphas_cumprod, quantize=True)
    self.sigma_data = .5

  def get_v(self, sample: FloatTensor, timestep: LongTensor) -> FloatTensor:
    out: UNet2DOutput = self.inner_model.forward(
      sample.to(self.inner_model.dtype),
      timestep,
    )
    return out.sample.to(self.sampling_dtype).neg()

  def forward(self, input, sigma, **kwargs) -> FloatTensor:
    nominal: FloatTensor = super().forward(input, sigma, **kwargs)
    # OpenAI applies static thresholding after scaling
    return nominal.clamp(-1, 1)