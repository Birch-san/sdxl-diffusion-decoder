import torch
from torch import inference_mode, FloatTensor, ByteTensor
from torchvision.io import read_image, write_png
from diffusers import ConsistencyDecoderVAE
from diffusers.models.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution, DecoderOutput
from k_diffusion.sampling import sample_euler_ancestral
from typing import Literal

from sdxl_diff_dec.sd_denoiser import SDDecoder

impl: Literal['diffusers-gan', 'diffusers-diffusion', 'kdiff-diffusion', 'openai-diffusion'] = 'openai-diffusion'

device = torch.device('cuda')
vae_sd: AutoencoderKL = AutoencoderKL.from_pretrained(
  'stabilityai/sd-vae-ft-mse',
  torch_dtype=torch.float16,
)
if impl != 'diffusers-gan':
  del vae_sd.decoder
vae_sd.to(device).eval()

in_img_path = 'in/5.jpg'
out_img_path = 'out/5.oai.png'
in_img: ByteTensor = read_image(in_img_path).to(dtype=vae_sd.dtype, device=device)
in_img_norm: FloatTensor = (in_img/127.5)-1.
# add batch dim
in_img_norm.unsqueeze_(0)

rng = torch.Generator()
with inference_mode():
  encoded: AutoencoderKLOutput = vae_sd.encode(in_img_norm.to(vae_sd.dtype))
enc_latent_dist: DiagonalGaussianDistribution = encoded.latent_dist
enc_latents: FloatTensor = enc_latent_dist.sample(generator=rng)

if impl != 'diffusers-gan':
  del vae_sd
if impl in ['diffusers-diffusion', 'kdiff-diffusion']:
  decoder: ConsistencyDecoderVAE = ConsistencyDecoderVAE.from_pretrained(
    'openai/consistency-decoder',
    variant='fp16',
    torch_dtype=torch.float16,
  ).to(device).eval()
elif impl == 'openai-diffusion':
  from consistencydecoder import ConsistencyDecoder
  decoder = ConsistencyDecoder(device=device)

if impl == 'kdiff-diffusion':
  denoiser = SDDecoder(decoder.decoder_unet, dtype=torch.float32)

# with autocast(dtype=torch.bfloat16):
with inference_mode():
  if impl == 'kdiff-diffusion':
    vae_scale_factor: int = 1 << (len(decoder.config.block_out_channels) - 1)
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
  else:
    decoded: DecoderOutput = decoder.decode(enc_latents)
    sample: FloatTensor = decoded.sample
write_png(sample.squeeze().clamp(-1, 1).add(1).mul(127.5).byte().cpu(), out_img_path)
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