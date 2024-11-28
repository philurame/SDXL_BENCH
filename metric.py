import torch, os
from tqdm import tqdm

from utils.class_registry import ClassRegistry
metrics_registry = ClassRegistry()

#####################################################################################################
# LPIPS, IS, FID, CLIP
#####################################################################################################
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class RequiredDataMixin:
  def __init__(self, path_ddim, fake_imgs, real_imgs, real_anns,
               pipe, nfe):
    self.path_ddim = path_ddim
    self.fake_imgs = fake_imgs
    self.real_imgs = real_imgs
    self.real_anns = real_anns
    self.pipe = pipe
    self.nfe = nfe

@metrics_registry.add_to_registry('LPIPS')
class LPIPSMetric(LearnedPerceptualImagePatchSimilarity, RequiredDataMixin):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    super().__init__(net_type='alex', reduction='mean', normalize=True)
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose
    self.device = device

  @torch.inference_mode()   
  def __call__(self, batch_size=512):
    self.reset()
    if os.path.exists(self.path_ddim):
      ddim_imgs = torch.load(self.path_ddim, map_location='cpu')
      for i in tqdm(range(0, self.fake_imgs.shape[0], batch_size), desc='LPIPS...', disable=not self.verbose):
        self.lpips.update(self.fake_imgs[i:i+batch_size].to(self.device, dtype=torch.float16)/255,
                          ddim_imgs[i:i+batch_size].to(self.device, dtype=torch.float16)/255)
      return self.compute().item()
    return None

@metrics_registry.add_to_registry('IS')
class ISMetric(InceptionScore, RequiredDataMixin):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    super().__init__(normalize=False)
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose
    self.device = device

  @torch.inference_mode()
  def __call__(self, fake_imgs, batch_size=512):
    self.reset()
    for i in tqdm(range(0, fake_imgs.shape[0], batch_size), desc='IS...', disable=not self.verbose):
      self.update(fake_imgs[i:i+batch_size].to(self.device))
    return self.compute()[0].item()
  
@metrics_registry.add_to_registry('FID')
class FIDMetric(FrechetInceptionDistance):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    super().__init__(feature=2048, normalize=False, reset_real_features=True)
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose
    self.device = device
  
  @torch.inference_mode()
  def __call__(self, batch_size=512):
    self.reset()
    for i in tqdm(range(0, self.fake_imgs.shape[0], batch_size), desc='FID...', disable=not self.verbose):
      self.update(self.real_imgs[i:i+batch_size].to(self.device), real=True)
      self.update(self.fake_imgs[i:i+batch_size].to(self.device), real=False)
    return self.compute().item()

@metrics_registry.add_to_registry('CLIP')
class CLIPMetric(CLIPScore, RequiredDataMixin):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    super().__init__(model_name_or_path="openai/clip-vit-base-patch32")
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose
    self.device = device

  @torch.inference_mode()
  def __call__(self, batch_size=512):
    self.reset()
    for i in tqdm(range(0, self.fake_imgs.shape[0], batch_size), desc='CLIP...', disable=not self.verbose):
      self.clip.update(self.fake_imgs[i:i+batch_size].to(self.device), self.real_anns[i:i+batch_size])
    return self.compute().item()

#####################################################################################################
# TFLOPS & SEC_PER_IMG
#####################################################################################################
from torch.profiler import profile, record_function, ProfilerActivity
import time

@metrics_registry.add_to_registry('TFLOPS')
class TFLOPSMetric(RequiredDataMixin):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose

  @torch.inference_mode()
  def __call__(self):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self.pipe(self.bench_prompt, 
             num_inference_steps=self.nfe, 
             guidance_scale=5, 
             return_dict=False, 
             output_type='latent')
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return total_flops/1e12

@metrics_registry.add_to_registry('SEC_PER_IMG')
class SEC_PER_IMG(RequiredDataMixin):
  def __init__(self, verbose=True, device='cpu', **data_kwargs):
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.verbose = verbose

  @torch.inference_mode()
  def __call__(self, warmup=20, num_gens=50):
    time_total = 0
    n_counted  = 0
    for i in tqdm(range(warmup + num_gens), desc='SEC_PER_IMG...', disable=not self.verbose):
      start_time = time.perf_counter()
      self.pipe(self.bench_prompt, 
             num_inference_steps=self.nfe, 
             guidance_scale=self.guidance_scale, 
             generator = torch.Generator(device='cuda').manual_seed(i),
             return_dict=False, 
             output_type='latent')
      torch.cuda.synchronize()
      if i >= warmup:
        time_total += time.perf_counter() - start_time
        n_counted += 1
    return time_total/n_counted