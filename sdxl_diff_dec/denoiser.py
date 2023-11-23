from k_diffusion.external import DiscreteVDDPMDenoiser
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from torch import FloatTensor, LongTensor
import torch

class DiffusersDenoiser(DiscreteVDDPMDenoiser):
  inner_model: UNet2DModel
  sigmas: FloatTensor
  log_sigmas: FloatTensor
  def __init__(self, unet: UNet2DModel, alphas_cumprod: FloatTensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_v(self, sample: FloatTensor, timestep: LongTensor) -> FloatTensor:
    out: UNet2DOutput = self.inner_model.forward(
      sample.to(self.inner_model.dtype),
      timestep,
    )
    return out.sample.to(self.sampling_dtype)