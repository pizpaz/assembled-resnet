from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from enum import Enum

Pooling = namedtuple('pooling', 'method until_block')
class PoolingMethod(Enum):
  max = 'max'
  avg = 'avg'
  stride = 'stride'
  none = 'none'

Branch = namedtuple('branch', 'method')
class BranchMethod(Enum):
  regular = 'regular'
  wide_and_deep = 'wide_and_deep'

class PaddingType(Enum):
  same = 'same'
  valid = 'valid'


class BlockLayerConfig(object):
  def __init__(self, output_channels, block_size, downsampling_method, padding_type):
    self.output_channels = output_channels
    self.block_size = block_size
    self.downsampling_method = downsampling_method
    self.padding_type = padding_type
    print('@@@ {}, {}, {}'.format(output_channels, block_size, downsampling_method, padding_type))

class BlockLayersConfig(object):
  def __init__(self, inputs):
    self.block_layer_configs = []
    print('@@@ BlockLayersConfig input = {}'.format(inputs))
    #piecewise_lr_boundary_epochs = [int(be) for be in inputs]
    for input in inputs:
      sp = input.split(':')
      assert(len(sp) == 6)
      output_channels = [int(c) for c in sp[:3]]
      block_size = int(sp[3])
      downsampling_method = sp[4]
      padding_type = sp[5]
      print('@@output={}, bs={}, dm={}, pt={}'.format(output_channels, block_size, downsampling_method, padding_type))
      block_layer_config = BlockLayerConfig(output_channels, block_size, PoolingMethod(downsampling_method),
                                            PaddingType(padding_type))
      self.block_layer_configs.append(block_layer_config)

      # for (output_channels, block_size, downsampling_method, padding_type) in inputs:
  def __iter__(self):
    return iter(self.block_layer_configs)
