import wandb, click
import torch, os, sys
from PIL import Image

from generate_decode import generate, decode_vae

from solver_scheduler import solver_registry, scheduler_registry
from cacher_quantizer import cacher_quantizer_registry

from data import data_registry
from metric import metrics_registry

##########################################################################################
##########################################################################################

def load_pipe(solver, scheduler, cacher_quantizer, optimize=False):
  PipeClass = cacher_quantizer_registry[cacher_quantizer]
  SolverClass = solver_registry[solver]
  SchedulerClassMixin = scheduler_registry[scheduler]
  
  class SolverSchedulerConstructor(SolverClass, SchedulerClassMixin):
    def set_timesteps(self, *args, **kwargs):
      SchedulerClassMixin.set_timesteps(self, *args, **kwargs)
  
  pipe = PipeClass.from_pretrained()
  pipe.scheduler = SolverSchedulerConstructor()

  pipe = pipe.to('cuda')
  pipe.set_progress_bar_config(disable=True)  

  # torch compile !cp /usr/include/crypt.h /home/ekneudachina/.conda/envs/philurame_venv/include/python3.9/
  if optimize:
    pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode='reduce-overhead', fullgraph=True)
  return pipe

def load_data(data_path, dataset, max_samples):
  return data_registry[dataset](data_path, max_samples)

def get_metrics(**kwargs):
  res_metrics = {}
  for metric_name in metrics_registry:
    metricInstance = metrics_registry[metric_name](**kwargs)
    res_metrics[metric_name] = metricInstance()
  return res_metrics

@click.command()
@click.option('--key', type=str, required=True, help='wandb key')
@click.option('--root', type=str, required=True, help='root dir full path')
@click.option('--max_samples', type=int, required=True, help='max num of images to generate')
@click.option('--dataset', type=str, required=True, help='PARTI | COCO')
@click.option('--generate', type=int, required=True, help='generate or calc metrics')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--solver', type=str, required=True, help='supported methods are in pipe_method.py')
@click.option('--scheduler', type=str, required=True, help='supported methods are in pipe_method.py')
@click.option('--cacher_quantizer', type=str, required=True, help='supported methods are in pipe_method.py')
def main(**kwargs):
  key = kwargs['key']
  root = kwargs['root']
  max_samples = kwargs['max_samples'] 
  dataset = kwargs['dataset']
  is_generate = kwargs['generate']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  cacher_quantizer = kwargs['cacher_quantizer']

  data_path = os.path.join(root, 'DATA')
  save_path = os.path.join(data_path, cacher_quantizer, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  print('\n'+'#'*50, 
        *[f'{k}={v}' for k, v in kwargs.items()],
        f'{save_path=}',
        '#'*50+'\n', sep='\n')
  sys.stdout.flush()

  assert dataset in ['PARTI', 'COCO']
  assert isinstance(nfe, int) and nfe > 0
  assert os.path.exists(root)
  assert bool(is_generate) != os.path.exists(save_path)
  assert 0 < max_samples <= 10000

  data = load_data(data_path, dataset, max_samples)
  pipe = load_pipe(solver, scheduler, cacher_quantizer, optimize=is_generate)

  is_metrics = not is_generate

  if is_generate: 
    ########################################
    # GENERAtE
    ########################################

    os.makedirs(os.path.dirname(save_path), exist_ok=False)
    gen_latents = generate(pipe, data.anns, nfe)
    torch.save(gen_latents, save_path)

    if solver == 'DDIM':
      gen_imgs = decode_vae(pipe, gen_latents)
      del gen_latents
      torch.save(gen_imgs, save_path)
  
  elif is_metrics:
    ########################################
    # METRICS
    ########################################

    is_already_decoded = torch.load(save_path, mmap=True).shape[1:] == (3, 1024, 1024)
    if is_already_decoded:
      gen_imgs = torch.load(save_path, map_location='cpu') # too large for cuda
    else: 
      gen_latents = torch.load(save_path, map_location='cuda')
      gen_imgs = decode_vae(pipe, gen_latents)
      del gen_latents

    path_ddim = os.path.join(data_path, cacher_quantizer, f'{dataset}_{nfe}', f'DDIM_{scheduler}.pt')
    metrics = get_metrics(
      path_ddim=path_ddim, 
      fake_imgs=gen_latents, 
      real_imgs=data.imgs, 
      real_anns=data.anns,
      pipe=pipe, 
      nfe=nfe
      )

    ########################################
    # LOG
    ########################################

    wandb.login(key=key, relogin=True)
    run = wandb.init(
      project = 'SDXL_METRICS_UPDATE',
      entity  = "philurame",
      name = f'{dataset}_{cacher_quantizer}_{solver}_{scheduler}_{nfe}',
      config = kwargs,
      save_code = True
    )

    # add image to metrics
    caption = data.anns[302 if dataset == 'PARTI' else 5449]
    gen_img  = wandb.Image(
      Image.fromarray(gen_imgs[302 if dataset == 'PARTI' else 5449].permute(1, 2, 0).numpy()),
      caption = caption)
    real_img = None if dataset == 'PARTI' else wandb.Image(
      Image.fromarray(data.imgs[5449].permute(1, 2, 0).numpy()),
      caption = caption)
    metrics['gen_img']  = gen_img
    metrics['real_img'] = real_img

    wandb.log(metrics)
    wandb.finish()


if __name__ == '__main__':
  main()