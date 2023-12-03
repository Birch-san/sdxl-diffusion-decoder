from diffusers.models.unet_2d import UNet2DModel
from torch.nn import Conv2d

def prune_conv_out(unet: UNet2DModel) -> None:
  '''
  Removes the 3 extraneous channels output by the SD consistency decoder UNet's final conv_out.
  They were probably variacne or something, useful during training but not any more.
  '''
  conv_out = unet.conv_out
  conv_out_lite = Conv2d(conv_out.in_channels, 3, kernel_size=conv_out.kernel_size, padding=conv_out.padding, device=conv_out.weight.device, dtype=conv_out.weight.dtype)
  conv_out_lite.weight.data.copy_(conv_out.weight[:3,:,:,:])
  conv_out_lite.bias.data.copy_(conv_out.bias[:3])

  unet.conv_out = conv_out_lite