from webdataset import WebDataset
from torch.utils.data import DataLoader
from torch import distributed as dist, multiprocessing as mp, FloatTensor
from typing import TypedDict, NotRequired, List, Optional
from accelerate import Accelerator
import torch
import argparse
from io import BytesIO
from tqdm import tqdm
import math

Sample = TypedDict('Sample', {
  '__key__': str,
  '__url__': str,
  'cls.txt': NotRequired[bytes],
  'img.png': bytes,
  'latent.pth': bytes,
})


def ensure_distributed():
  if not dist.is_initialized():
    dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def collate_fn(samples: List[Sample]) -> FloatTensor:
  latents: List[FloatTensor] = []
  for sample in samples:
    latent_data: bytes = sample['latent.pth']
    with BytesIO(latent_data) as stream:
      latent_t: FloatTensor = torch.load(stream, weights_only=True)
    latents.append(latent_t)
  latents_batch: FloatTensor = torch.stack(latents)
  return latents_batch

def main():
  p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--in-pattern', type=str, required=True,
                  help='wds pattern')
  p.add_argument('--dataset-len', type=int, default=None,
                  help='estimate of dataset length, for progress bar')
  p.add_argument('--batch-size', type=int, default=128,
                  help='the batch size')
  p.add_argument('--num-workers', type=int, default=8,
                  help='the number of data loader workers')
  p.add_argument('--start-method', type=str, default='fork',
                  choices=['fork', 'forkserver', 'spawn'],
                  help='the multiprocessing start method')

  args = p.parse_args()
  mp.set_start_method(args.start_method)
  torch.backends.cuda.matmul.allow_tf32 = True

  accelerator = Accelerator()
  ensure_distributed()

  print(f'Verifying {args.in_pattern}')
  dataset = WebDataset(args.in_pattern)
  if args.dataset_len is None:
    batches_estimate: Optional[int] = None
  else:
    batches_estimate: int = math.ceil(args.dataset_len/(args.batch_size*accelerator.num_processes))

  dl = DataLoader(dataset, args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
  dl = accelerator.prepare(dl)

  for batch_ix, latents in enumerate(tqdm(dl, 'verifying', total=batches_estimate, unit='batch')):
    all_latents: FloatTensor = accelerator.gather(latents)
    assert not all_latents.isnan().any().item(), f'NaN found in batch {batch_ix}'
    assert not all_latents.isinf().any().item(), f'Inf found in batch {batch_ix}'
  print(f'Finished verifying {args.in_pattern}')

if __name__ == '__main__':
  main()