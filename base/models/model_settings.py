import os.path

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')

MODEL_POOL = {
    'st2f256': {
        'resolution': 256,
        'repeat_w': False,
        'final_tanh': True,
        'use_bn': True,
    },
    'ganinv_bedroom256': {
        'resolution': 256,
        'repeat_w': False,
        'final_tanh': True,
        'use_bn': True,
    },
    'ganinv_tower256': {
        'resolution': 256,
        'repeat_w': False,
        'final_tanh': True,
        'use_bn': True,
    },
}

# Settings for GAN.
GAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
GAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
GAN_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 2

MAX_IMAGES_ON_RAM = 800


def get_weight_path(weight_name):

  assert isinstance(weight_name, str)
  if weight_name == '':
    return ''
  if weight_name[-4:] != '.pth':
    weight_name += '.pth'
  return os.path.join(MODEL_DIR, weight_name)
