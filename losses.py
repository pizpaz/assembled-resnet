from absl import logging
from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.framework import ops

class LossAndAccuracy:
  def __init__(self):
    return

  @abstractmethod
  def loss_and_update_acc(self, labels, logits, training):
    raise NotImplementedError('Must be implemented in subclasses.')


class CategoricalCrossEntropyAndAcc(LossAndAccuracy):
  def __init__(self, batch_size, num_classes, label_smoothing):
    super(CategoricalCrossEntropyAndAcc, self).__init__()
    logging.info('CategoricalCrossEntropyAndAcc: num_classes({}), label_smoothing({})'.format(num_classes,
                                                                                              label_smoothing))
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.label_smoothing = label_smoothing
    with ops.init_scope():
      self.training_loss = 0.
      self.test_loss = 0.

    if label_smoothing > 0.0:
      self.training_accuracy = tf.keras.metrics.CategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
      self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    else:
      self.training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
      self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'test_accuracy', dtype=tf.float32)


  def loss_and_update_acc(self, labels, logits, training):
    if self.label_smoothing > 0.0:
      labels = tf.squeeze(labels)
      onehot_labels = tf.one_hot(labels, self.num_classes, dtype=tf.float32, axis=-1)
      loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits,
                                                      label_smoothing=self.label_smoothing)
    else:
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    loss = tf.reduce_sum(loss) * (1.0 / self.batch_size)

    if training:
      self.training_loss = loss
      self.training_accuracy.update_state(labels, logits)
    else:
      self.test_loss = loss
      self.test_accuracy.update_state(labels, logits)

    return loss

  def loss_and_update_acc_onehot(self, labels, onehot_labels, logits, training):
    logging.info('labels=({}), onehot_labels=({}), logits=({})'.format(labels, onehot_labels, logits))
    loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits,
                                                    label_smoothing=self.label_smoothing)
    loss = tf.reduce_sum(loss) * (1.0 / self.batch_size)

    if training:
      self.training_loss = loss
      self.training_accuracy.update_state(onehot_labels, logits)
    else:
      self.test_loss = loss
      self.test_accuracy.update_state(labels, logits)

    return loss
