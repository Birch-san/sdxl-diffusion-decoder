import torch.utils.data as data
from pathlib import Path
from torch import nn
from PIL import Image

# from k-diffusion
# https://github.com/crowsonkb/k-diffusion/blob/045515774882014cc14c1ba2668ab5bad9cbf7c0/k_diffusion/utils.py#L387
# By Katherine Crowson
# MIT-licensed
# modified to return a PIL Image instead of a 1-tuple
class FolderOfImages(data.Dataset):
  """Recursively finds all images in a directory. It does not support
  classes/targets."""

  IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

  def __init__(self, root, transform=None):
    super().__init__()
    self.root = Path(root)
    self.transform = nn.Identity() if transform is None else transform
    self.paths = sorted(path for path in self.root.rglob('*') if path.suffix.lower() in self.IMG_EXTENSIONS)

  def __repr__(self):
    return f'FolderOfImages(root="{self.root}", len: {len(self)})'

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, key) -> Image.Image:
    path = self.paths[key]
    with open(path, 'rb') as f:
      image = Image.open(f).convert('RGB')
    image = self.transform(image)
    return image