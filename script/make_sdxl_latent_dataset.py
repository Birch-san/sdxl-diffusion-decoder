import accelerate
import argparse
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution
import torch
from torch import distributed as dist, multiprocessing as mp
from torch import FloatTensor, IntTensor, inference_mode
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import math
from typing import Dict, Any, Iterator, Optional, Callable, TypedDict, List, NamedTuple, Union
from pathlib import Path
from welford_torch import Welford
from tqdm import tqdm
from os import makedirs
from contextlib import nullcontext
from PIL import Image
from dataclasses import dataclass, field

from sdxl_diff_dec.vae.attn.natten_attn_processor import NattenAttnProcessor
from sdxl_diff_dec.vae.attn.null_attn_processor import NullAttnProcessor
from sdxl_diff_dec.vae.attn.qkv_fusion import fuse_vae_qkv
from sdxl_diff_dec.dataset.folder_of_images import FolderOfImages

SinkOutput = TypedDict('SinkOutput', {
  '__key__': str,
  'img.png': Image.Image,
  'latent.pth': FloatTensor,
})

HasClassTxt = TypedDict('HasClassTxt', {
  'cls.txt': str,
})

class ClassySinkOutput(SinkOutput, HasClassTxt): pass

class LabeledImg(NamedTuple):
  orig_img: Image.Image
  cls: int

class LabeledImgAndOrig(NamedTuple):
  pil_img: Image.Image
  img_t: FloatTensor
  cls: int

class LabeledImgAndOrigBatch(NamedTuple):
  pil_imgs: List[Image.Image]
  img_t: FloatTensor
  classes: IntTensor

class ImgAndOrig(NamedTuple):
  pil_img: Image.Image
  img_t: FloatTensor

class ImgAndOrigBatch(NamedTuple):
  pil_imgs: List[Image.Image]
  img_t: FloatTensor

@dataclass
class ClassyImageDatasetWithOrig(torch.utils.data.Dataset):
  dataset: datasets.ImageFolder
  transform: Optional[Callable[[Image.Image], Image.Image]] = field(default=None)
  to_tensor: transforms.ToTensor = field(init=False, default_factory=transforms.ToTensor)

  def __getitem__(self, index: int) -> LabeledImgAndOrig:
    item: LabeledImg = self.dataset[index]
    pil_img, cls = item
    pil_img: Image.Image = pil_img if self.transform is None else self.transform(pil_img)
    img_t: FloatTensor = self.to_tensor(pil_img)
    return LabeledImgAndOrig(pil_img, img_t, cls)

  def __len__(self):
    return len(self.dataset)

@dataclass
class ImageDatasetWithOrig(torch.utils.data.Dataset):
  dataset: FolderOfImages
  transform: Optional[Callable[[Image.Image], Image.Image]] = field(default=None)
  to_tensor: transforms.ToTensor = field(init=False, default_factory=transforms.ToTensor)

  def __getitem__(self, index: int) -> ImgAndOrig:
    pil_img: Image.Image = self.dataset[index]
    pil_img: Image.Image = pil_img if self.transform is None else self.transform(pil_img)
    img_t: FloatTensor = self.to_tensor(pil_img)
    return ImgAndOrig(pil_img, img_t)

  def __len__(self):
    return len(self.dataset)

def custom_collate_classy(batch: List[LabeledImgAndOrig]) -> LabeledImgAndOrigBatch:
  transposed = list(zip(*batch))
  pil_imgs, img_t, labels = transposed
  img_t = torch.utils.data.dataloader.default_collate(img_t)
  labels = torch.utils.data.dataloader.default_collate(labels)
  return LabeledImgAndOrigBatch(pil_imgs, img_t, labels)

def custom_collate(batch: List[ImgAndOrig]) -> ImgAndOrigBatch:
  transposed = list(zip(*batch))
  pil_imgs, img_t = transposed
  img_t = torch.utils.data.dataloader.default_collate(img_t)
  return ImgAndOrigBatch(pil_imgs, img_t)

def ensure_distributed():
  if not dist.is_initialized():
    dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def main():
  p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--in-dir', type=str, required=True,
                  help='root of a folder-of-images dataset (with subdirectories for classes)')
  p.add_argument('--batch-size', type=int, default=4,
                  help='the batch size')
  p.add_argument('--num-workers', type=int, default=8,
                  help='the number of data loader workers')
  p.add_argument('--side-length', type=int, default=None,
                  help='square side length to which to resize-and-crop image. if None: we will assume the dataset is already at the desired size.')
  p.add_argument('--seed', type=int,
                  help='the random seed')
  p.add_argument('--log-average-every-n', type=int, default=1000,
                  help='how noisy do you want your logs to be (log the online average per-channel mean and std of latents every n batches)')
  p.add_argument('--save-average-every-n', type=int, default=10000,
                  help="how frequently to save the welford averages. the main reason we do it on an interval is just so there's no nasty surprise at the end of the run.")
  p.add_argument('--use-ollin-vae', action='store_true',
                  help="use Ollin's fp16 finetune of SDXL 0.9 VAE")
  p.add_argument('--compile', action='store_true',
                  help="accelerate VAE with torch.compile()")
  p.add_argument('--has-class-dirs', action='store_true',
                  help="use torchvision.datasets.ImageFolder to traverse an image folder dataset which has class subdirectoriees")
  p.add_argument('--start-method', type=str, default='fork',
                  choices=['fork', 'forkserver', 'spawn'],
                  help='the multiprocessing start method')
  p.add_argument('--vae-attn-impl', type=str, default='original',
                  choices=['original', 'natten', 'null'],
                  help='use more lightweight attention in VAE (https://github.com/Birch-san/sdxl-play/pull/3)')
  p.add_argument('--out-root', type=str, default="./out/latents",
                  help='directory into which to output WDS .tar files')

  args = p.parse_args()
  mp.set_start_method(args.start_method)
  torch.backends.cuda.matmul.allow_tf32 = True
  try:
    torch._dynamo.config.automatic_dynamic_shapes = False
  except AttributeError:
    pass

  accelerator = accelerate.Accelerator()
  ensure_distributed()
  assert accelerator.num_processes == 1, "I just noticed that this won't work multi-process because you can't gather a List[Image.Image]. the workaround would be to gather the original images as tensors, but then the main process would be forced to either write original images uncompressed into the wds, or re-encode the images to PNG for writing into dataset. probably simplest recourse is just to run single-GPU."

  if args.seed is not None:
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
    torch.manual_seed(seeds[accelerator.process_index])
  latent_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())

  tf = None if args.side_length is None else transforms.Compose(
    [
      transforms.Resize(args.side_length, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
      transforms.CenterCrop(args.side_length),
    ]
  )
  inner_dataset: Union[datasets.ImageFolder, FolderOfImages] = datasets.ImageFolder(args.in_dir) if args.has_class_dirs else FolderOfImages(args.in_dir)

  dataset = ClassyImageDatasetWithOrig(inner_dataset, transform=tf) if args.has_class_dirs else ImageDatasetWithOrig(inner_dataset, transform=tf)
  dataset_len: int = len(dataset)
  batches_estimate: int = math.ceil(dataset_len/(args.batch_size*accelerator.num_processes))

  collate_fn = custom_collate_classy if args.has_class_dirs else custom_collate
  dl = DataLoader(dataset, args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)

  vae_kwargs: Dict[str, Any] = {
    'torch_dtype': torch.float16,
  } if args.use_ollin_vae else {
    'subfolder': 'vae',
    'torch_dtype': torch.bfloat16,
  }
  vae: AutoencoderKL = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix' if args.use_ollin_vae else 'stabilityai/stable-diffusion-xl-base-0.9',
    use_safetensors=True,
    **vae_kwargs,
  )

  if args.vae_attn_impl == 'natten':
    fuse_vae_qkv(vae)
    # NATTEN seems to output identical output to global self-attention at kernel size 17
    # even kernel size 3 looks good (not identical, but very close).
    # I haven't checked what's the smallest kernel size that can look identical. 15 looked good too.
    # seems to speed up encoding of 1024x1024px images by 11%
    vae.set_attn_processor(NattenAttnProcessor(kernel_size=17))
  elif args.vae_attn_impl == 'null':
    for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
      # you won't be needing these
      del attn.to_q, attn.to_k
    # doesn't mix information between tokens via QK similarity. just projects every token by V and O weights.
    # looks alright, but is by no means identical to global self-attn.
    vae.set_attn_processor(NullAttnProcessor())
  elif args.vae_attn_impl == 'original':
    # leave it as global self-attention
    pass
  else:
    raise ValueError(f"Never heard of --vae-attn-impl '{args.vae_attn_impl}'")

  del vae.decoder
  vae.to(accelerator.device).eval()
  if args.compile:
    vae = torch.compile(vae, fullgraph=True, mode='max-autotune')

  dl = accelerator.prepare(dl)

  wds_out_dir = Path(args.out_root) / 'wds'
  avg_out_dir = Path(args.out_root) / 'avg'

  if accelerator.is_main_process:
    from webdataset import ShardWriter
    makedirs(wds_out_dir, exist_ok=True)
    makedirs(avg_out_dir, exist_ok=True)
    sink_ctx = ShardWriter(f'{wds_out_dir}/%05d.tar', maxcount=10000)
    w_val = Welford(device=accelerator.device)
    w_sq = Welford(device=accelerator.device)
    def make_base_sample(ix: int, pil_img: Image.Image, latent: FloatTensor) -> SinkOutput:
      out: SinkOutput = {
        '__key__': str(ix),
        'img.png': pil_img,
        'latent.pth': latent,
      }
      return out
    if args.has_class_dirs:
      def make_classy_sample(base: SinkOutput, cls: int) -> ClassySinkOutput:
        return {
          **base,
          '__key__': f'{cls}/{base["__key__"]}',
          'cls.txt': str(cls),
        }
      def make_sample(ix: int, pil_img: Image.Image, latent: FloatTensor, cls: Optional[int]) -> ClassySinkOutput:
        base: SinkOutput = make_base_sample(ix, pil_img, latent)
        classy: ClassySinkOutput = make_classy_sample(base, cls)
        return classy
    else:
      make_sample: Callable[[int, Image.Image, FloatTensor, int], SinkOutput] = lambda ix, pil_img, latent, _: make_base_sample(ix, pil_img, latent)

    def sink_sample(sink: ShardWriter, ix: int, pil_img: Image.Image, latent: FloatTensor, cls: Optional[int]) -> None:
      out: Union[SinkOutput, ClassySinkOutput] = make_sample(ix, pil_img, latent, cls)
      sink.write(out)
  else:
    sink_ctx = nullcontext()
    sink_sample: Callable[[Optional[ShardWriter], int, Image.Image, FloatTensor, Optional[int]], None] = lambda *_: ...
    w_val: Optional[Welford] = None
    w_sq: Optional[Welford] = None

  samples_output = 0
  it: Iterator[Union[LabeledImgAndOrigBatch, ImgAndOrigBatch]] = iter(dl)
  with sink_ctx as sink:
    for batch_ix, batch in enumerate(tqdm(it, total=batches_estimate)):
      if isinstance(batch, LabeledImgAndOrigBatch):
        pil_imgs, img_t, classes = batch
      elif isinstance(batch, ImgAndOrigBatch):
        pil_imgs, img_t = batch
        classes: Optional[IntTensor] = None
      else:
        raise ValueError('batch was neither LabeledImgAndOrigBatch nor ImgAndOrigBatch')
      img_t = img_t.to(vae.dtype)

      # for sample_ix, sample in enumerate(img_t.unbind()):
      #   save_image(sample.unsqueeze(0), f'out/vae-decode-test/b{batch_ix}_s{sample_ix}.orig.png')

      # transform pipeline's ToTensor() gave us [0, 1]
      # but VAE wants [-1, 1]
      img_t.mul_(2).sub_(1)
      with inference_mode():
        encoded: AutoencoderKLOutput = vae.encode(img_t)
      dist: DiagonalGaussianDistribution = encoded.latent_dist
      latents: FloatTensor = dist.sample(generator=latent_gen)
      all_latents: FloatTensor = accelerator.gather(latents)

      # you can verify correctness by saving the sample out like so:
      # from torchvision.utils import save_image
      # with inference_mode():
      #   all_rgbs: FloatTensor = vae.decode(all_latents).sample.div(2).add_(.5).clamp_(0,1)
      # for sample_ix, sample in enumerate(all_rgbs.unbind()):
      #   save_image(sample.unsqueeze(0), f'out/vae-decode-test/b{batch_ix}_s{sample_ix}.roundtrip.pre.png')

      # let's not multiply by scale factor. opt instead to measure a per-channel scale-and-shift
      #   all_latents.mul_(vae.config.scaling_factor)

      if accelerator.is_main_process:
        per_channel_val_mean: FloatTensor = all_latents.mean((-1, -2))
        per_channel_sq_mean: FloatTensor = all_latents.square().mean((-1, -2))
        w_val.add_all(per_channel_val_mean)
        w_sq.add_all(per_channel_sq_mean)

        if batch_ix % args.log_average_every_n == 0:
          print('per-channel val:', w_val.mean)
          print('per-channel  sq:', w_sq.mean)
          print('per-channel std:', torch.sqrt(w_sq.mean - w_val.mean**2))
        if batch_ix % args.save_average_every_n == 0:
          print(f'Saving averages to {avg_out_dir}')
          torch.save(w_val.mean, f'{avg_out_dir}/val.pt')
          torch.save(w_sq.mean,  f'{avg_out_dir}/sq.pt')
        del per_channel_val_mean, per_channel_sq_mean

      if isinstance(batch, LabeledImgAndOrigBatch):
        all_classes: FloatTensor = accelerator.gather(classes)
        classes_unbound: List[FloatTensor] = all_classes.unbind()
      else:
        all_classes: Optional[FloatTensor] = None
        classes_unbound: List[Optional[FloatTensor]] = [None]*all_latents.shape[0]
      for sample_ix_in_batch, (pil_img, latent, cls) in enumerate(zip(pil_imgs, all_latents.unbind(), classes_unbound)):
        sample_ix_in_corpus: int = batch_ix * args.batch_size * accelerator.num_processes + sample_ix_in_batch
        if sample_ix_in_corpus >= dataset_len:
          break
        cls_item: Optional[int] = None if cls is None else cls.item()
        # it's crucial to transfer the _sample_ to CPU, not the batch. otherwise each sample we serialize, has the whole batch's data hanging off it
        sink_sample(sink, sample_ix_in_corpus, pil_img, latent.cpu(), cls_item)
        samples_output += 1
      del all_latents, latents, all_classes, classes, classes_unbound
  print(f"r{accelerator.process_index} done")
  if accelerator.is_main_process:
    print(f'Output {samples_output} samples. We wanted {dataset_len}.')
    print('per-channel val:', w_val.mean)
    print('per-channel  sq:', w_sq.mean)
    print('per-channel std:', torch.sqrt(w_sq.mean - w_val.mean**2))
    print(f'Saving averages to {avg_out_dir}')
    torch.save(w_val.mean, f'{avg_out_dir}/val.pt')
    torch.save(w_sq.mean,  f'{avg_out_dir}/sq.pt')

if __name__ == '__main__':
  main()