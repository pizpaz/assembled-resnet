from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Default(object):
  shuffle_buffer = 1000
  train_image_size = 224
  num_channels = 3
  num_images = {
    'train': 0,
    'validation': 0
  }
  num_train_files = 128

class ImageNet(Default):
  shuffle_buffer = 10000
  num_classes = 1001
  num_images = {
    'train': 1281167,
    'validation': 50000,
  }
  num_train_files = 1024
  dataset_name = 'imagenet'

class Food101(Default):
  shuffle_buffer = 1000
  num_classes = 101
  num_images = {
    'train': 75750,
    'validation': 25250,
  }
  num_train_files = 128
  dataset_name = 'food101'

def get_config(data_name):
  if data_name == 'imagenet':
    return ImageNet()
  elif data_name == 'food101':
    return Food101()
  else:
    raise ValueError("Unable to support {} dataset.".format(data_name))
