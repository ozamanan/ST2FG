import os.path
import sys
import logging
import numpy as np

import torch

from . import model_settings

__all__ = ['BaseModule']

DTYPE_NAME_TO_TORCH_TENSOR_TYPE = {
    'float16': torch.HalfTensor,
    'float32': torch.FloatTensor,
    'float64': torch.DoubleTensor,
    'int8': torch.CharTensor,
    'int16': torch.ShortTensor,
    'int32': torch.IntTensor,
    'int64': torch.LongTensor,
    'uint8': torch.ByteTensor,
    'bool': torch.BoolTensor,
}


def get_temp_logger(logger_name='logger'):
  if not logger_name:
    raise ValueError(f'Input `logger_name` should not be empty!')

  logger = logging.getLogger(logger_name)
  if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

  return logger


class BaseModule(object):
  def __init__(self, model_name, module_name, logger=None):
    self.model_name = model_name
    self.module_name = module_name
    self.logger = logger or get_temp_logger(model_name)

    # Parse settings.
    for key, val in model_settings.MODEL_POOL[model_name].items():
      setattr(self, key, val)
    self.use_cuda = model_settings.USE_CUDA and torch.cuda.is_available()
    self.batch_size = model_settings.MAX_IMAGES_ON_DEVICE
    self.ram_size = model_settings.MAX_IMAGES_ON_RAM
    self.net = None
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    # Check necessary settings.
    self.check_attr('gan_type')  # Should be specified in derived classes.
    self.check_attr('resolution')
    self.image_channels = getattr(self, 'image_channels', 3)
    assert self.image_channels in [1, 3]
    self.channel_order = getattr(self, 'channel_order', 'RGB').upper()
    assert self.channel_order in ['RGB', 'BGR']
    self.min_val = getattr(self, 'min_val', -1.0)
    self.max_val = getattr(self, 'max_val', 1.0)

    # Get paths.
    self.weight_path = model_settings.get_weight_path(
        f'{model_name}_{module_name}')

    # Build graph and load pre-trained weights.
    self.logger.info(f'Build network for module `{self.module_name}` in '
                     f'model `{self.model_name}`.')
    self.model_specific_vars = []
    self.build()
    if os.path.isfile(self.weight_path):
      self.load()
    else:
      self.logger.warning(f'No pre-trained weights will be loaded!')

    # Change to inference mode and GPU mode if needed.
    assert self.net
    self.net.eval().to(self.run_device)

  def check_attr(self, attr_name):
    if not hasattr(self, attr_name):
      raise AttributeError(f'Field `{attr_name}` is missing for '
                           f'module `{self.module_name}` in '
                           f'model `{self.model_name}`!')

  def build(self):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def load(self):
    self.logger.info(f'Loading pytorch weights from `{self.weight_path}`.')
    state_dict = torch.load(self.weight_path)
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.net.state_dict()[var_name]
    self.net.load_state_dict(state_dict)
    self.logger.info(f'Successfully loaded!')

  def to_tensor(self, array):
    dtype = type(array)
    if isinstance(array, torch.Tensor):
      tensor = array
    elif isinstance(array, np.ndarray):
      tensor_type = DTYPE_NAME_TO_TORCH_TENSOR_TYPE[array.dtype.name]
      tensor = torch.from_numpy(array).type(tensor_type)
    else:
      raise ValueError(f'Unsupported input type `{dtype}`!')
    tensor = tensor.to(self.run_device)
    return tensor

  def get_value(self, tensor):
    dtype = type(tensor)
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{dtype}`!')

  def get_ont_hot_labels(self, num, labels=None):
    self.check_attr('label_size')
    if self.label_size == 0:
      return None

    if labels is None:
      labels = 0
    labels = np.array(labels).reshape(-1)
    if labels.size == 1:
      labels = np.tile(labels, (num,))
    assert labels.shape == (num,)
    for label in labels:
      if label >= self.label_size or label < 0:
        raise ValueError(f'Label should be smaller than {self.label_size}, '
                         f'but {label} is received!')

    one_hot = np.zeros((num, self.label_size), dtype=np.int32)
    one_hot[np.arange(num), labels] = 1
    return one_hot

  def get_batch_inputs(self, inputs, batch_size=None):
    total_num = inputs.shape[0]
    batch_size = batch_size or self.batch_size
    for i in range(0, total_num, batch_size):
      yield inputs[i:i + batch_size]

  def batch_run(self, inputs, run_fn):
    if inputs.shape[0] > self.ram_size:
      self.logger.warning(f'Number of inputs on RAM is larger than '
                          f'{self.ram_size}. Please use '
                          f'`self.get_batch_inputs()` to split the inputs! '
                          f'Otherwise, it may encounter OOM problem!')

    results = {}
    temp_key = '__temp_key__'
    for batch_inputs in self.get_batch_inputs(inputs):
      batch_outputs = run_fn(batch_inputs)
      if isinstance(batch_outputs, dict):
        for key, val in batch_outputs.items():
          if not isinstance(val, np.ndarray):
            raise ValueError(f'Each item of the model output should be with '
                             f'type `numpy.ndarray`, but type `{type(val)}` is '
                             f'received for key `{key}`!')
          if key not in results:
            results[key] = [val]
          else:
            results[key].append(val)
      elif isinstance(batch_outputs, np.ndarray):
        if temp_key not in results:
          results[temp_key] = [batch_outputs]
        else:
          results[temp_key].append(batch_outputs)
      else:
        raise ValueError(f'The model output can only be with type '
                         f'`numpy.ndarray`, or a dictionary of '
                         f'`numpy.ndarray`, but type `{type(batch_outputs)}` '
                         f'is received!')

    for key, val in results.items():
      results[key] = np.concatenate(val, axis=0)
    return results if temp_key not in results else results[temp_key]
