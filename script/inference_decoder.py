import torch
from torch import inference_mode, FloatTensor, ByteTensor
from torchvision.io import read_image, write_png
from diffusers import ConsistencyDecoderVAE
from diffusers.models.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution, DecoderOutput
# from torch.cuda.amp import autocast
import gc

device = torch.device('cuda')
vae_sd: AutoencoderKL = AutoencoderKL.from_pretrained(
  'stabilityai/sd-vae-ft-mse',
  torch_dtype=torch.float16,
)
del vae_sd.decoder
vae_sd.to(device).eval()

in_img_path = 'out/vae-decode-test2/b0_s5.orig.png'
out_img_path = 'out/vae-decode-test2/b0_s5.oai.png'
# in_img_path = 'out/vae-decode-test/b0_s4.orig.png'
# out_img_path = 'out/vae-decode-test/b0_s4.oai.png'
in_img: ByteTensor = read_image(in_img_path).to(dtype=vae_sd.dtype, device=device)
in_img_norm: FloatTensor = (in_img/127.5)-1.
# add batch dim
in_img_norm.unsqueeze_(0)

rng = torch.Generator()
with inference_mode():
  encoded: AutoencoderKLOutput = vae_sd.encode(in_img_norm.to(vae_sd.dtype))
enc_latent_dist: DiagonalGaussianDistribution = encoded.latent_dist
enc_latents: FloatTensor = enc_latent_dist.sample(generator=rng)

del vae_sd
gc.collect()
torch.cuda.empty_cache()
decoder: ConsistencyDecoderVAE = ConsistencyDecoderVAE.from_pretrained(
  'openai/consistency-decoder',
  variant='fp16',
  torch_dtype=torch.float16,
).to(device).eval()

# normalize to standard Gaussian (what the UNet expects)
# enc_latents.mul_(vae_sd.config.scaling_factor)

with inference_mode():
# with autocast(dtype=torch.bfloat16):
  decoded: DecoderOutput = decoder.decode(enc_latents)
sample: FloatTensor = decoded.sample
write_png(decoded.sample.squeeze().clamp(-1, 1).add(1).mul(127.5).byte().cpu(), out_img_path)
pass