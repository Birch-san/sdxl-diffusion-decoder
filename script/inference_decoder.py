import torch
from torch import inference_mode, FloatTensor, ByteTensor
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from os.path import isfile
from os import getenv, makedirs
from shutil import copyfile
from pathlib import Path
from diffusers import ConsistencyDecoderVAE
from diffusers.models.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution, DecoderOutput
from k_diffusion.sampling import BrownianTreeNoiseSampler
from typing import Literal, Dict, List, Optional

from sdxl_diff_dec.sd_denoiser import SDDecoderDistilled
from sdxl_diff_dec.normalize import Normalize
from sdxl_diff_dec.schedule import betas_for_alpha_bar, alpha_bar, get_alphas
from sdxl_diff_dec.sampling import sample_consistency

seed = 42

device = torch.device('cuda')
ImplType = Literal['diffusers-gan', 'diffusers-diffusion', 'kdiff-diffusion', 'openai-diffusion']
impl: ImplType = 'kdiff-diffusion'
# disabled because it's encountering weird artifacting
kdiff_use_brownian_tree = False

out_qualifiers: Dict[ImplType, str] = {
  'diffusers-gan': 'diffgan',
  'diffusers-diffusion': 'diff2',
  'kdiff-diffusion': 'kdiff',
  'openai-diffusion': 'oai',
}
out_qualifier: str = out_qualifiers[impl]

out_dir = Path('out')
makedirs(out_dir, exist_ok=True)

in_img_path = Path(f'in/gt2.png')
out_img_path: Path = out_dir / f'{in_img_path.stem}.{out_qualifier}.png'
cached_enc_latents_path: Path = out_dir / f'{in_img_path.stem}.cache.pt'

enc_latents_found: bool = isfile(cached_enc_latents_path)
if enc_latents_found:
  print(f'Found precomputed latents at {cached_enc_latents_path}')
  enc_latents: FloatTensor = torch.load(cached_enc_latents_path, map_location=device, weights_only=True)

convenience_copy = out_dir / in_img_path.name
if not isfile(convenience_copy):
  copyfile(in_img_path, convenience_copy)

if impl == 'diffusers-gan' or not enc_latents_found:
  vae_sd: AutoencoderKL = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-mse',
    torch_dtype=torch.float16,
  )
  if impl != 'diffusers-gan':
    del vae_sd.decoder
  vae_sd.to(device).eval()

if not enc_latents_found:
  in_img: ByteTensor = read_image(str(in_img_path)).to(dtype=vae_sd.dtype, device=device)
  in_img_norm: FloatTensor = (in_img/127.5)-1.
  del in_img
  # add batch dim
  in_img_norm.unsqueeze_(0)

  rng_latent_samp = torch.Generator().manual_seed(seed)
  with inference_mode():
    encoded: AutoencoderKLOutput = vae_sd.encode(in_img_norm.to(vae_sd.dtype))
    del in_img_norm
  enc_latent_dist: DiagonalGaussianDistribution = encoded.latent_dist
  enc_latents: FloatTensor = enc_latent_dist.sample(generator=rng_latent_samp)
  torch.save(enc_latents, cached_enc_latents_path)
  del enc_latent_dist, encoded, vae_sd.encoder
  if impl != 'diffusers-gan':
    del vae_sd

if impl in ['diffusers-diffusion', 'kdiff-diffusion']:
  cdecoder: ConsistencyDecoderVAE = ConsistencyDecoderVAE.from_pretrained(
    'openai/consistency-decoder',
    variant='fp16',
    torch_dtype=torch.float16,
  ).to(device).eval()
elif impl == 'openai-diffusion':
  from consistencydecoder import ConsistencyDecoder
  openai_decoder = ConsistencyDecoder(device=device, download_root=getenv('OPENAI_CACHE_DIR', '~/.cache/clip'))

if impl == 'kdiff-diffusion':
  total_timesteps=1024
  betas = betas_for_alpha_bar(total_timesteps, alpha_bar, device=device)
  alphas: FloatTensor = get_alphas(betas)
  alphas_cumprod: FloatTensor = alphas.cumprod(dim=0)

  denoiser = SDDecoderDistilled(
    cdecoder.decoder_unet,
    alphas_cumprod,
    total_timesteps=total_timesteps,
    n_distilled_steps=64,
    dtype=torch.float32,
  )

  # divided by 0.18215, so we don't have to do OpenAI's bizarre "standardize the wonky way, then scale-and-shift from there"
  channel_means: List[float] = [2.1335418224334717, 0.12369272112846375, 0.4052227735519409, -0.09404008090496063]
  channel_stds: List[float] = [5.300093650817871, 5.731559753417969, 4.180506229400635, 4.228494644165039]
  normalize = Normalize(channel_means, channel_stds).to(device)

rng = torch.Generator().manual_seed(seed)
steps = 2

with inference_mode():
  if impl == 'kdiff-diffusion':
    # scale-and-shift latents from VAE distribution to standard Gaussian
    normalize.forward_(enc_latents)

    vae_scale_factor: int = 1 << (len(cdecoder.config.block_out_channels) - 1)
    enc_latents = F.interpolate(enc_latents, mode="nearest", scale_factor=vae_scale_factor)

    B, _, H, W = enc_latents.shape
    # note: seems to tolerate 2-step sampling, but something's definitely wrong with step counts higher than that.
    sigmas: FloatTensor = denoiser.get_sigmas_rounded(n=steps+1, include_sigma_min=False)
    if kdiff_use_brownian_tree:
      # seems to be exhibiting odd artifacting. perhaps it's set up incorrectly.
      noise_sampler = BrownianTreeNoiseSampler(
        torch.empty((B, 3, H, W), device=device),
        sigma_min=sigmas[-2],
        sigma_max=sigmas[0],
        seed=seed,
        transform=lambda sigma: denoiser.sigma_to_t(sigma),
      )
      noise: FloatTensor = noise_sampler(sigmas[0], sigmas[1])
    else:
      noise: FloatTensor = torch.randn((B, 3, H, W), generator=rng).to(device)
      noise_sampler: Optional[BrownianTreeNoiseSampler] = None
    noise.mul_(sigmas[0])
    extra_args={'latents': enc_latents}
    sample: FloatTensor = sample_consistency(denoiser, noise, sigmas, extra_args=extra_args, noise_sampler=noise_sampler)
  elif impl == 'openai-diffusion':
    torch.manual_seed(seed)
    sample: FloatTensor = openai_decoder.__call__(enc_latents, schedule=torch.linspace(1, 0, steps+1)[:-1].tolist())
  elif impl == 'diffusers-diffusion':
    decoded: DecoderOutput = cdecoder.decode(enc_latents, num_inference_steps=steps, generator=rng)
    sample: FloatTensor = decoded.sample
  elif impl == 'diffusers-gan':
    decoded: DecoderOutput = vae_sd.decode(enc_latents)
    sample: FloatTensor = decoded.sample
  else:
    raise ValueError(f'Unrecognised decoder implementation "{impl}"')
write_png(sample.squeeze().clamp(-1, 1).add(1).mul(127.5).byte().cpu(), str(out_img_path))
print(f'Saved {out_img_path}')
pass

# they take your encoded latents,
# assume you already scaled them, and undo it.
# scale-and-shift,
# 8x nearest-upsample,
# draw 3-channel noise at the 8x size,
# multiply the noise by
#   self.decoder_scheduler.init_noise_sigma
# which is 0.9997,
# start sampling from timestep 1008,
# whose self.c_in[timestep] is 1.002
# c_in is a ramp from 2 to 1 in 1024 steps
# they multiply the noise by that,
# concat z onto it.