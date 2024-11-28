import torch
import torch.nn as nn
import re

def seed_everything(seed=42):
  import random, torch
  import numpy as np
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

# linear only
LINEAR_LAYER_ONLY_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|proj_(in|out))$"
# ALL \ time embed
DEFAULT_LAYER_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|conv(\d+|_shortcut)?|proj_(in|out))$"

def get_linear_and_conv_order(pipe, refilter=None, min_channels=16):
  if refilter is None:
    refilter = LINEAR_LAYER_ONLY_REGEX

  def layer_filter_fn(layer: nn.Module, layer_name: str) -> bool:
    if isinstance(layer, (nn.Conv2d)):
      if min(layer.in_channels, layer.out_channels) < min_channels:
        return
    elif isinstance(layer, nn.Linear):
      if min(layer.in_features, layer.out_features) < min_channels:
        return
    else:
      return
    return re.search(refilter, layer_name)

  # collect groups
  down_group = []
  # collect from down blocks
  for i, block in enumerate(pipe.unet.down_blocks):
    group = set()
    for module_name, module in block.named_modules():
      full_module_name = f"down_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group.add((full_module_name, module))
    down_group.append(list(group))

  # collect from mid block
  group = set()
  block = pipe.unet.mid_block
  for module_name, module in block.named_modules():
    full_module_name = f"mid_block.{module_name}"
    if layer_filter_fn(module, full_module_name):
      group.add((full_module_name, module))
  mid_group = [list(group)]

  up_group = []
  # collect from up blocks
  for i, block in enumerate(pipe.unet.up_blocks):
    group = set()
    for module_name, module in block.named_modules():
      full_module_name = f"up_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group.add((full_module_name, module))
    up_group.append(list(group))
  all_layers = down_group+mid_group+up_group
  all_layers = [i[1] for j in all_layers for i in j]
  return all_layers