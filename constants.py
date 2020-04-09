from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from enum import Enum

Pooling = namedtuple('pooling', 'method until_block')
class PoolingMethod(Enum):
  max = 'max'
  avg = 'avg'
  none = 'none'