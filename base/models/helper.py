from .model_settings import MODEL_POOL
from .gan_generator import GANGenerator
from .gan_encoder import GANEncoder
from .perceptual_model import PerceptualModel

__all__ = ['build_generator', 'build_encoder', 'build_perceptual']


def build_generator(model_name, logger=None):
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type in ['ganinv']:
    return GANGenerator(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_encoder(model_name, logger=None):
  if model_name not in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = model_name.split('_')[0]
  if gan_type == 'ganinv':
    return GANEncoder(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


build_perceptual = PerceptualModel
