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

  def shortcut(self, x, num_filters, strides, name):
    logging.info('@@@shorcut ON!')
    x = layers.AveragePooling2D(pool_size=(2,2), strides=strides,
                                padding='same', data_format=self.image_data_format)(
                                x)
    x = layers.Conv2D(
      num_filters, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=resnet_model.gen_l2_regularizer(self.use_l2_regularizer),
      name=name + 'd1')(
      x)

    return x

###TEMP CODE
#### TODO: refer to https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/pooling.py#L282-L328
#### we should rewrite this applying keras style.
def generalized_mean_pooling(x, p=3, data_format='channels_first'):
  if data_format == 'channels_first':
    _, c, h, w = x.shape.as_list()
    reduce_axis = [2, 3]
  else:
    _, h, w, c = x.shape.as_list()
    reduce_axis = [1, 2]

  N = tf.to_float(tf.multiply(h, w))
  if x.dtype == tf.float16:
    # 수치 안정성을 위해 fp16 시 fp32로 캐스팅 후 계산 후, 다시 fp16으로 바꾼다.
    x = tf.cast(x, tf.float32)

  epsilon = 1e-6
  x = tf.clip_by_value(x, epsilon, 1e12)
  x_p = tf.pow(x, p)
  x_p_sum = tf.maximum(tf.reduce_sum(x_p, axis=reduce_axis, keep_dims=True), epsilon)
  pooled_x = tf.pow(N, -1.0 / p) * tf.pow(x_p_sum, 1 / p)
  if x.dtype == tf.float16:
    pooled_x = tf.cast(pooled_x, tf.float16)
  return pooled_x