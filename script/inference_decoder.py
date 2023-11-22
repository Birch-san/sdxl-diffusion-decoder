import torch
from torch import inference_mode, FloatTensor, ByteTensor
# from torch.cuda.amp import autocast
from torchvision.io import read_image, write_png
from os.path import isfile
from os import getenv, makedirs
from shutil import copyfile
from pathlib import Path
from diffusers import ConsistencyDecoderVAE
from diffusers.models.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution, DecoderOutput
from k_diffusion.sampling import sample_euler_ancestral
from typing import Literal, Dict

from sdxl_diff_dec.sd_denoiser import SDDecoder

seed = 42

device = torch.device('cuda')
ImplType = Literal['diffusers-gan', 'diffusers-diffusion', 'kdiff-diffusion', 'openai-diffusion']
impl: ImplType = 'diffusers-gan'

out_qualifiers: Dict[ImplType, str] = {
  'diffusers-gan': 'diffgan',
  'diffusers-diffusion': 'diff2',
  'kdiff-diffusion': 'kdiff',
  'openai-diffusion': 'oai',
}
out_qualifier: str = out_qualifiers[impl]

out_dir = Path('out')
makedirs(out_dir, exist_ok=True)

in_img_path = Path(f'in/5.jpg')
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
  in_img: ByteTensor = read_image(in_img_path).to(dtype=vae_sd.dtype, device=device)
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
  denoiser = SDDecoder(cdecoder.decoder_unet, dtype=torch.float32)

rng = torch.Generator().manual_seed(seed)

# with autocast(dtype=torch.bfloat16):
with inference_mode():
  if impl == 'kdiff-diffusion':
    vae_scale_factor: int = 1 << (len(cdecoder.config.block_out_channels) - 1)
    noise = torch.randn(
      (1, 3, vae_scale_factor*enc_latents.shape[-2], vae_scale_factor*enc_latents.shape[-1]),
      generator=rng,
    ).to(device)
    sigmas: FloatTensor = denoiser.get_sigmas(2)
    noise.mul_(sigmas[0])

    # decrease variance of latents to make them more like a standard Gaussian
    enc_latents.mul_(vae_sd.config.scaling_factor)
    # per-channel scale-and-shift to be *even more* like a standard Gaussian
    denoiser.normalize.forward_(enc_latents)

    extra_args={'latents': enc_latents}
    sample: FloatTensor = sample_euler_ancestral(denoiser, noise, sigmas, extra_args=extra_args)
  elif impl == 'openai-diffusion':
    sample: FloatTensor = openai_decoder.__call__(enc_latents)
  elif impl == 'diffusers-diffusion':
    decoded: DecoderOutput = cdecoder.decode(enc_latents)
    sample: FloatTensor = decoded.sample
  elif impl == 'diffusers-gan':
    decoded: DecoderOutput = vae_sd.decode(enc_latents)
    sample: FloatTensor = decoded.sample
  else:
    raise ValueError(f'Unrecognised decoder implementation "{impl}"')
write_png(sample.squeeze().clamp(-1, 1).add(1).mul(127.5).byte().cpu(), str(out_img_path))
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