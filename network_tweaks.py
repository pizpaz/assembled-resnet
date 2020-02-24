from absl import logging

from tensorflow.python.keras import layers as tf_python_keras_layers

import resnet_model

layers = tf_python_keras_layers

class NetworkTweaks:
  def __init__(self):
    return

  def getConfig(self):
    return


class ResnetD(NetworkTweaks):
  def __init__(self, image_data_format, use_l2_regularizer):
    super(ResnetD, self).__init__()
    logging.info("ResnetD ON")
    self.image_data_format = image_data_format
    self.use_l2_regularizer = use_l2_regularizer

  def input(self, x, num_filters):
    if self.image_data_format == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1

    x = layers.Conv2D(
      num_filters // 2, (3, 3),
      strides=(2, 2),
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=resnet_model.gen_l2_regularizer(self.use_l2_regularizer),
      name='dconv1')(
      x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=resnet_model.BATCH_NORM_DECAY,
      epsilon=resnet_model.BATCH_NORM_EPSILON,
      fused=True,
      name='bn_dconv1')(
      x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
      num_filters // 2, (3, 3),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=resnet_model.gen_l2_regularizer(self.use_l2_regularizer),
      name='dconv2')(
      x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=resnet_model.BATCH_NORM_DECAY,
      epsilon=resnet_model.BATCH_NORM_EPSILON,
      fused=True,
      name='bn_dconv2')(
      x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
      num_filters, (3, 3),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=resnet_model.gen_l2_regularizer(self.use_l2_regularizer),
      name='dconv3')(
      x)

    return x

  def shortcut(self, x, num_filters, strides):
    logging.info('@@@shorcut ON!')
    x = layers.AveragePooling2D(pool_size=(2,2), strides=strides,
                                padding='same', data_format=self.image_data_format)(
                                x)
    x = layers.Conv2D(
      num_filters, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=resnet_model.gen_l2_regularizer(self.use_l2_regularizer))(
      x)

    return x
