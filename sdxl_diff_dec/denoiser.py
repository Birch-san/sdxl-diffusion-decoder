from k_diffusion.external import DiscreteEpsDDPMDenoiser
from diffusers import UNet2DConditionModel, UNet2DConditionOutput
from torch import FloatTensor, LongTensor
import torch

class DiffusersDenoiser(DiscreteEpsDDPMDenoiser):
  inner_model: UNet2DConditionModel
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: FloatTensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_eps(self, sample: FloatTensor, timestep: LongTensor) -> FloatTensor:
    out: UNet2DConditionOutput = self.inner_model.forward(
      sample.to(self.inner_model.dtype),
      timestep,
    )
    return out.sample.to(self.sampling_dtype)