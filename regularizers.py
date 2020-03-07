from absl import logging

import tensorflow as tf
import tensorflow_probability as tfp
import common

tfd = tfp.distributions

class Mixup(common.TrainIteration):
  def __init__(self, alpha, flags_obj):
    super(Mixup, self).__init__(flags_obj)
    self.alpha=alpha

  def get_num_train_iterations(self):
    train_steps, train_epochs, eval_steps = super(Mixup, self).get_num_train_iterations()
    return train_steps*2, train_epochs*2, eval_steps

  def __call__(self, x, y):
    logging.info('@@MIXUP! ({})'.format(self.alpha))
    dist = tfd.Beta(self.alpha, self.alpha)

    _, h, w, c = x.get_shape().as_list()

    batch_size = tf.shape(x)[0]
    num_class = y.get_shape().as_list()[1]

    lam1 = dist.sample([batch_size // 2])

    if x.dtype == tf.float16:
      lam1 = tf.cast(lam1, dtype=tf.float16)
      y = tf.cast(y, dtype=tf.float16)

    x1, x2 = tf.split(x, 2, axis=0)
    y1, y2 = tf.split(y, 2, axis=0)

    lam1_x = tf.tile(tf.reshape(lam1, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
    lam1_y = tf.tile(tf.reshape(lam1, [batch_size // 2, 1]), [1, num_class])

    mixed_sx1 = lam1_x * x1 + (1. - lam1_x) * x2
    mixed_sy1 = lam1_y * y1 + (1. - lam1_y) * y2
    mixed_sx1 = tf.stop_gradient(mixed_sx1)
    mixed_sy1 = tf.stop_gradient(mixed_sy1)

    logging.info('mixup images=({}), mixup labels=({})'.format(mixed_sx1, mixed_sy1))

    return mixed_sx1, mixed_sy1
