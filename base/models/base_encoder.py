import numpy as np

from .base_module import BaseModule

__all__ = ['BaseEncoder']


class BaseEncoder(BaseModule):
  def __init__(self, model_name, logger=None):
    self.encode_dim = None  # Target shape of the encoded code.
    super().__init__(model_name, 'encoder', logger)
    assert self.encode_dim is not None
    assert isinstance(self.encode_dim, (list, tuple))

  def preprocess(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')

    if images.ndim != 4 or images.shape[3] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, height, width '
                       f'channel], where channel equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    if images.shape[3] == 1 and self.image_channels == 3:
      images = np.tile(images, (1, 1, 1, 3))
    if images.shape[3] != self.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{images.shape[3]}, is not supported by the current '
                       f'encoder, which requires {self.image_channels} '
                       f'channels!')
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images.astype(np.float32)
    images = images / 255.0 * (self.max_val - self.min_val) + self.min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)

    return images

  def encode(self, images, **kwargs):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_encode(self, images, **kwargs):
    return self.encode(self.preprocess(images), **kwargs)
