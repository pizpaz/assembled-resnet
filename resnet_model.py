# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers as tf_python_keras_layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
import imagenet_preprocessing
import constants

L2_WEIGHT_DECAY = 1e-4
#BATCH_NORM_DECAY = 0.9
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5

STRIDE_SIZE=(2,2)

layers = tf_python_keras_layers


def change_keras_layer(use_tf_keras_layers=False):
  """Change layers to either tf.keras.layers or tf.python.keras.layers.

  Layer version of  tf.keras.layers is depends on tensorflow version, but
  tf.python.keras.layers checks environment variable TF2_BEHAVIOR.
  This function is a temporal function to use tf.keras.layers.
  Currently, tf v2 batchnorm layer is slower than tf v1 batchnorm layer.
  this function is useful for tracking benchmark result for each version.
  This function will be removed when we use tf.keras.layers as default.

  TODO(b/146939027): Remove this function when tf v2 batchnorm reaches training
  speed parity with tf v1 batchnorm.

  Args:
      use_tf_keras_layers: whether to use tf.keras.layers.
  """
  global layers
  if use_tf_keras_layers:
    layers = tf.keras.layers
  else:
    layers = tf_python_keras_layers


def gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   zero_gamma=False,
                   use_l2_regularizer=True):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    zero_gamma: Initialize γ = 0 for all BN layers that sit at the end of a residual block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  if zero_gamma:
    gamma_initializer = 'zeros'
  else:
    gamma_initializer = 'ones'
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      gamma_initializer=gamma_initializer,
      fused=True,
      name=bn_name_base + '2c')(
          x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               zero_gamma=False,
               use_l2_regularizer=True,
               resnetd=None,
               pooling_method=None,
               padding_type=None):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    zero_gamma: Initialize γ = 0 for all BN layers that sit at the end of a residual block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    resnetd:
    pooling_method:
    padding_type:

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  if pooling_method == constants.PoolingMethod.max:
    logging.info('@MAX pooling at stage {}'.format(stage))
    x = layers.MaxPooling2D(pool_size=STRIDE_SIZE)(x)
    conv_strides = (1,1)
  elif pooling_method == constants.PoolingMethod.avg:
    logging.info('@AVG pooling at stage {}'.format(stage))
    x = layers.AveragePooling2D(pool_size=STRIDE_SIZE)(x)
    conv_strides = (1,1)
  elif pooling_method == constants.PoolingMethod.stride:
    logging.info('@Stride conv at stage {}'.format(stage))
    conv_strides = STRIDE_SIZE
  elif pooling_method == constants.PoolingMethod.none:
    logging.info('@No downsampling at stage {}'.format(stage))
    conv_strides = (1,1)
  else:
    raise NotImplementedError

  x = layers.Conv2D(
    filters2,
    kernel_size,
    strides=conv_strides,
    padding=padding_type.value,
    use_bias=False,
    kernel_initializer='he_normal',
    kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
    name=conv_name_base + '2b')(
    x)

  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  if zero_gamma:
    gamma_initializer = 'zeros'
  else:
    gamma_initializer = 'ones'
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      gamma_initializer=gamma_initializer,
      fused=True,
      name=bn_name_base + '2c')(
          x)

  if resnetd is None:
    shortcut = layers.Conv2D(
        filters3,
        kernel_size=(1,1) if padding_type == constants.PaddingType.same else kernel_size,
        strides=(1,1) if pooling_method == constants.PoolingMethod.none else STRIDE_SIZE,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(
            input_tensor)
  else:
    shortcut = resnetd.shortcut(x=input_tensor, num_filters=filters3,
                                strides=(1, 1) if pooling_method == constants.PoolingMethod.none else STRIDE_SIZE,
                                name=conv_name_base)

  shortcut = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name=bn_name_base + '1')(
          shortcut)

  logging.info('@@ X = {}, shortcut = {}'.format(x, shortcut))
  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def resnet50(num_classes,
             train_image_size,
             batch_size=None,
             zero_gamma=False,
             last_pool_channel_type='gap',
             use_l2_regularizer=True,
             rescale_inputs=False,
             resnetd=None,
             pooling=None,
             include_top=True,
             branch=None):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.
    train_image_size: TODO
    batch_size: Size of the batches for each step.
    zero_gamma: Initialize γ = 0 for all BN layers that sit at the end of a residual block.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    pooling: namedtuple. Pooling(method, until_block)

  Returns:
      A Keras model instance.
  """
  input_shape = (train_image_size, train_image_size, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
        lambda x: x * 255.0 - backend.constant(
            imagenet_preprocessing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype),
        name='rescale')(
            img_input)
  else:
    x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Lambda(
        lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
        name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  if resnetd is None:
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
  else:
    x = resnetd.input(x, 64)
  logging.info("@first = {}".format(x))

  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      fused=True,
      name='bn_conv1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
  logging.info("@2 = {}".format(x))

  x = conv_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='a',
      strides=(1, 1),
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer,
      resnetd=resnetd)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='b',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='c',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  logging.info("@3 = {}".format(x))

  x = conv_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='a',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer,
      resnetd=resnetd,
      pooling_method=pooling.method if pooling.until_block >= 3 else None)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='b',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='c',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='d',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  logging.info("@4 = {}".format(x))

  x = conv_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='a',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer,
      resnetd=resnetd,
      pooling_method = pooling.method if pooling.until_block >= 4 else None)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='b',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='c',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='d',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='e',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='f',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  logging.info("@5 = {}".format(x))

  x = conv_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='a',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer,
      resnetd=resnetd,
      pooling_method = pooling.method if pooling.until_block >= 5 else None)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='b',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='c',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
  logging.info("@6 = {}".format(x))

  if last_pool_channel_type == 'gap':
    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
    logging.info("@gap = {}".format(x))
  elif last_pool_channel_type == 'gmp':
    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.max(x, rm_axes), name='reduce_max')(x)
    logging.info("@gmp = {}".format(x))
  else:
    pool_type, channel_size = last_pool_channel_type.split('_')
    channel_size = int(channel_size)
    (wa, ha) = (1, 2) if backend.image_data_format == 'channels_last' else (2, 3)
    xs = tf.shape(x)
    #channel_size가 64이면 최종 x는 32*7*7=1568, 32이면 64*7*7=3136
    x = tf.reshape(x, [xs[0], channel_size, -1, xs[wa], xs[ha]])
    if pool_type == 'mean':
      x = layers.Lambda(lambda x: backend.mean(x, [1]), name='reduce_mean')(x)
    elif pool_type == 'max':
      x = layers.Lambda(lambda x: backend.max(x, [1]), name='reduce_mean')(x)
    else:
      raise NotImplementedError
    x = layers.Flatten()(x)

  if include_top:
    logging.info('@@@include top!')
    x = layers.Dense(
      num_classes,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name='fc{}'.format(num_classes))(
          x)


    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)
  else:
    logging.info('@@not include top')

  # Create model.
  return models.Model(img_input, x, name='resnet50')

def resnet50_new(num_classes,
             train_image_size,
             block_layers_config,
             batch_size=None,
             zero_gamma=False,
             last_pool_channel_type='gap',
             use_l2_regularizer=True,
             rescale_inputs=False,
             resnetd=None,
             pooling=None,
             include_top=True,
             branch=None):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.
    train_image_size: TODO
    batch_size: Size of the batches for each step.
    zero_gamma: Initialize γ = 0 for all BN layers that sit at the end of a residual block.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    pooling: namedtuple. Pooling(method, until_block)

  Returns:
      A Keras model instance.
  """
  input_shape = (train_image_size, train_image_size, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
      lambda x: x * 255.0 - backend.constant(
        imagenet_preprocessing.CHANNEL_MEANS,
        shape=[1, 1, 3],
        dtype=x.dtype),
      name='rescale')(
      img_input)
  else:
    x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Lambda(
      lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
      name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  if resnetd is None:
    x = layers.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
      x)
  else:
    x = resnetd.input(x, 64)
  logging.info("@0 = {}".format(x))

  x = layers.BatchNormalization(
    axis=bn_axis,
    momentum=BATCH_NORM_DECAY,
    epsilon=BATCH_NORM_EPSILON,
    fused=True,
    name='bn_conv1')(
    x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
  logging.info("@1 = {}".format(x))

  for stage, block_layer_config in enumerate(block_layers_config):
    stage += 1
    x = conv_block(
      x,
      3, block_layer_config.output_channels,
      stage=stage,
      block='a',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer,
      resnetd=resnetd,
      pooling_method=block_layer_config.downsampling_method,
      padding_type=block_layer_config.padding_type)
    x = identity_block(
      x,
      3, block_layer_config.output_channels,
      stage=stage,
      block='b',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
      x,
      3, block_layer_config.output_channels,
      stage=stage,
      block='c',
      zero_gamma=zero_gamma,
      use_l2_regularizer=use_l2_regularizer)
    logging.info("@{} = {}".format(stage, x))

  if last_pool_channel_type == 'gap':
    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
    logging.info("@gap = {}".format(x))
  elif last_pool_channel_type == 'gmp':
    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.max(x, rm_axes), name='reduce_max')(x)
    logging.info("@gmp = {}".format(x))
  else:
    pool_type, channel_size = last_pool_channel_type.split('_')
    channel_size = int(channel_size)
    (wa, ha) = (1, 2) if backend.image_data_format == 'channels_last' else (2, 3)
    xs = tf.shape(x)
    #channel_size가 64이면 최종 x는 32*7*7=1568, 32이면 64*7*7=3136
    x = tf.reshape(x, [xs[0], channel_size, -1, xs[wa], xs[ha]])
    if pool_type == 'mean':
      x = layers.Lambda(lambda x: backend.mean(x, [1]), name='reduce_mean')(x)
    elif pool_type == 'max':
      x = layers.Lambda(lambda x: backend.max(x, [1]), name='reduce_mean')(x)
    else:
      raise NotImplementedError
    x = layers.Flatten()(x)

  if include_top:
    logging.info('@@@include top!')
    x = layers.Dense(
      num_classes,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=gen_l2_regularizer(use_l2_regularizer),
      name='fc{}'.format(num_classes))(
      x)


    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)
  else:
    logging.info('@@not include top')

  # Create model.
  return models.Model(img_input, x, name='resnet50')

def get_top_layer(x, num_classes, use_l2_regularizer):
  x = layers.Dense(
    num_classes,
    kernel_initializer=initializers.RandomNormal(stddev=0.01),
    kernel_regularizer=gen_l2_regularizer(use_l2_regularizer),
    bias_regularizer=gen_l2_regularizer(use_l2_regularizer),
    name = 'fc{}'.format(num_classes))(x)
  x = layers.Activation('softmax', dtype='float32')(x)

  return x
