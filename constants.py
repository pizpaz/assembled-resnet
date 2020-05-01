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

Branch = namedtuple('branch', 'method')
class BranchMethod(Enum):
  regular = 'regular'
  wide_and_deep = 'wide_and_deep'

