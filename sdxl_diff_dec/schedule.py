from torch import FloatTensor, tensor
import math
from typing import Callable

alpha_bar: Callable[[float], float] = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

def betas_for_alpha_bar(
  num_diffusion_timesteps: int,
  alpha_bar: Callable[[float], float],
  max_beta=0.999,
  device='cpu',
) -> FloatTensor:
  # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L45
  betas = []
  for i in range(num_diffusion_timesteps):
    t1 = i / num_diffusion_timesteps
    t2 = (i + 1) / num_diffusion_timesteps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
  return tensor(betas, device=device)

def get_alphas(betas: FloatTensor) -> FloatTensor:
  return 1.0 - betas