import wandb, os
from PIL import Image
import pandas as pd

def add_sample_imgs(dataset, data, gen_imgs):
  caption = data.anns[302 if dataset == 'PARTI' else 5449]
  gen_img  = wandb.Image(
    Image.fromarray(gen_imgs[302 if dataset == 'PARTI' else 5449].permute(1, 2, 0).numpy()),
    caption = caption)
  real_img = None if dataset == 'PARTI' else wandb.Image(
    Image.fromarray(data.imgs[5449].permute(1, 2, 0).numpy()),
    caption = caption)
  return dict(real_img = real_img, gen_img = gen_img)

def is_already_calculated(save_path, calculated_path):
  cacher_quantizer, dataset_nfe, solver_scheduler = save_path.split('/')[-3:]
  dataset, nfe = dataset_nfe.split('_')
  solver, scheduler = solver_scheduler.replace('.pt', '').split('_')
  if not os.path.exists(calculated_path):
    return True
  df_already_calculated = pd.read_csv(calculated_path, index_col=0)
  res = df_already_calculated.query(
    "dataset == @dataset and \
    solver == @solver and \
    scheduler == @scheduler and \
    cacher_quantizer == @cacher_quantizer and \
    nfe == @nfe"
    ).shape[0] == 0
  return res
