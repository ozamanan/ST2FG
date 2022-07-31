import numpy as np

from .base_module import BaseModule

__all__ = ['BaseGenerator']


class BaseGenerator(BaseModule):
  def __init__(self, model_name, logger=None):

    super().__init__(model_name, 'generator', logger)

  def sample(self, num, **kwargs):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def preprocess(self, latent_codes, **kwargs):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_sample(self, num, **kwargs):

    return self.preprocess(self.sample(num, **kwargs), **kwargs)

  def synthesize(self, latent_codes, **kwargs):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def postprocess(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    if images.ndim != 4 or images.shape[1] != self.image_channels:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images

  def easy_synthesize(self, latent_codes, **kwargs):

    outputs = self.synthesize(latent_codes, **kwargs)
    if 'image' in outputs:
      outputs['image'] = self.postprocess(outputs['image'])
    return outputs
