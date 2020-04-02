# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow.python.keras import models
#from tensorflow.python.ops import array_ops

import imagenet_preprocessing
import common
import resnet_model
import regularizers
import network_tweaks
import losses
import dataset_config

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers

flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                          'tf.function.')
flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                     help='Calculate L2_loss on concatenated weights, '
                          'instead of using Keras per-layer L2 loss.')

flags.DEFINE_float(name='base_learning_rate', short_name='blr', default=0.01,
                   help=flags_core.help_wrap('Base learning rate.'))
flags.DEFINE_boolean(name='zero_gamma', default=False,
                     help=flags_core.help_wrap(
                       'If True, we initialize gamma = 0 for all BN layers that sit at the end of a residual block'))
flags.DEFINE_string(name='autoaugment_type', default=None,
                    help=flags_core.help_wrap(
                    'Specifies auto augmentation type. One of "imagenet", "svhn", "cifar", "good"'
                    'To use numpy implementation, prefix "np_" to the type.'))
flags.DEFINE_string(name='learning_rate_decay_type', short_name='lrdt', default='exponential',
                    help=flags_core.help_wrap(
                    'Specifies how the learning rate is decayed. One of '
                    '"piecewise", "cosine"'))
#### Common
flags.DEFINE_string(
  name='dataset_name', default=None,
  help=flags_core.help_wrap('imagenet, food100, food101, naver_food547, cub_200_2011'))

#### Network Tweak
flags.DEFINE_boolean(name='use_resnet_d', default=False,
                     help=flags_core.help_wrap('Use resnet_d architecture. '
                                               'For more details, refer to https://arxiv.org/abs/1812.01187'))
flags.DEFINE_integer(name='max_pooling', default=0,
                     help=flags_core.help_wrap('Use max pooling instead of stride conv.'))

#### Regularization
flags.DEFINE_float(name='label_smoothing', short_name='lblsm', default=0.0,
                   help=flags_core.help_wrap('If greater than 0 then smooth the labels.'))
flags.DEFINE_integer(name='mixup_type', short_name='mixup_type', default=0,
                     help=flags_core.help_wrap(
                      'Use mixup data augmentation. For more details, refer to https://arxiv.org/abs/1710.09412'
                      '타입이 0이면, mixup을 사용하지 않는다.'
                      '타입이 1이면, batch_size의 두배를 mixup해서 batch_size만큼의 데이터를 만든다'))

##### Experimental
# 0=>gap
flags.DEFINE_string(name='last_pool_channel_type', default="gap",
                    help=flags_core.help_wrap(''))



def build_stats(train_result, eval_result, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    train_result: The final loss at training time.
    eval_result: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback instance.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}

  if eval_result:
    stats['eval_loss'] = eval_result[0]
    stats['eval_acc'] = eval_result[1]

    stats['train_loss'] = train_result[0]
    stats['train_acc'] = train_result[1]

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats['avg_exp_per_second'] = (
              time_callback.batch_size * time_callback.log_steps *
              (len(time_callback.timestamp_log) - 1) /
              (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats


def get_input_dataset(flags_obj, strategy, dataset_conf):
  """Returns the test and train input datasets."""
  dtype = flags_core.get_tf_dtype(flags_obj)
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  batch_size = flags_obj.batch_size
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
        'Batch size must be divisible by number of replicas : {}'.format(
          strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  if flags_obj.use_synthetic_data:
    input_fn = common.get_synth_input_fn(
      height=dataset_conf.default_image_size,
      width=dataset_conf.default_image_size,
      num_channels=dataset_conf.num_channels,
      num_classes=dataset_conf.num_classes,
      dtype=dtype,
      drop_remainder=True)
  else:
    input_fn = imagenet_preprocessing.input_fn

  def _train_dataset_fn(ctx=None):
    if flags_obj.mixup_type > 0:
      train_batch_size = batch_size*2
    else:
      train_batch_size = batch_size
    train_ds = input_fn(
        is_training=True,
        data_dir=flags_obj.data_dir,
        dataset_conf=dataset_conf,
        batch_size=train_batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        dtype=dtype,
        input_context=ctx,
        drop_remainder=True,
        autoaugment_type=flags_obj.autoaugment_type)
    return train_ds

  if strategy:
    if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
      train_ds = strategy.experimental_distribute_datasets_from_function(_train_dataset_fn)
    else:
      train_ds = strategy.experimental_distribute_dataset(_train_dataset_fn())
  else:
    train_ds = _train_dataset_fn()

  test_ds = None
  if not flags_obj.skip_eval:
    def _test_data_fn(ctx=None):
      test_ds = input_fn(
          is_training=False,
          data_dir=flags_obj.data_dir,
          dataset_conf=dataset_conf,
          batch_size=batch_size,
          parse_record_fn=imagenet_preprocessing.parse_record,
          dtype=dtype,
          input_context=ctx)
      return test_ds

    if strategy:
      if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
        test_ds = strategy.experimental_distribute_datasets_from_function(
            _test_data_fn)
      else:
        test_ds = strategy.experimental_distribute_dataset(_test_data_fn())
    else:
      test_ds = _test_data_fn()

  return train_ds, test_ds


'''
def get_num_train_iterations(flags_obj):
  """Returns the number of training steps, train and test epochs."""
  train_steps = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1

  eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  return train_steps, train_epochs, eval_steps
'''

def _steps_to_run(steps_in_current_epoch, steps_per_epoch, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  if steps_per_loop == 1:
    return steps_per_loop
  return min(steps_per_loop, steps_per_epoch - steps_in_current_epoch)


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using custom training loops.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  print('@@@@enable_eager = {}'.format(flags_obj.enable_eager))
  dataset_conf = dataset_config.get_config(flags_obj.dataset_name)
  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_float16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)
  elif dtype == tf.bfloat16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  # This only affects GPU.
  common.set_cudnn_batchnorm_mode()

  # TODO(anj-s): Set data_format without using Keras.
  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=distribution_utils.configure_cluster(flags_obj.worker_hosts, flags_obj.task_index),
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu)

  if flags_obj.mixup_type > 0:
    mixup = regularizers.Mixup(0.2, flags_obj)
    train_iteration = mixup
  else:
    mixup = None
    train_iteration = common.TrainIteration(flags_obj)

  train_ds, test_ds = get_input_dataset(flags_obj, strategy, dataset_conf)

  per_epoch_steps, train_epochs, eval_steps = train_iteration.get_num_train_iterations()
  steps_per_loop = min(flags_obj.steps_per_loop, per_epoch_steps)
  logging.info("Training %d epochs, each epoch has %d steps, "
               "total steps: %d; Eval %d steps",
               train_epochs, per_epoch_steps, train_epochs * per_epoch_steps,
               eval_steps)

  time_callback = keras_utils.TimeHistory(flags_obj.batch_size,
                                          flags_obj.log_steps)

  with distribution_utils.get_strategy_scope(strategy):
    resnet_model.change_keras_layer(flags_obj.use_tf_keras_layers)
    use_l2_regularizer = not flags_obj.single_l2_loss_op

    if flags_obj.use_resnet_d:
      resnetd = network_tweaks.ResnetD(image_data_format=tf.keras.backend.image_data_format(),
                                       use_l2_regularizer=use_l2_regularizer)
    else:
      resnetd = None

    model = resnet_model.resnet50(
        num_classes=dataset_conf.num_classes,
        batch_size=flags_obj.batch_size,
        zero_gamma=flags_obj.zero_gamma,
        last_pool_channel_type=flags_obj.last_pool_channel_type,
        use_l2_regularizer=use_l2_regularizer,
        resnetd=resnetd,
        max_pooling=flags_obj.max_pooling,
        include_top=True if flags_obj.pretrained_filepath == '' else False)

    if flags_obj.learning_rate_decay_type == 'piecewise':
        lr_schedule = common.PiecewiseConstantDecayWithWarmup(
            batch_size=flags_obj.batch_size,
            epoch_size=dataset_conf.num_images['train'],
            warmup_epochs=common.LR_SCHEDULE[0][1],
            boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
            multipliers=list(p[0] for p in common.LR_SCHEDULE),
            compute_lr_on_cpu=True)
    elif flags_obj.learning_rate_decay_type == 'cosine':
        lr_schedule = common.CosineDecayWithWarmup(
            base_lr=flags_obj.base_learning_rate,
            batch_size=flags_obj.batch_size,
            epoch_size=dataset_conf.num_images['train'],
            warmup_epochs=common.LR_SCHEDULE[0][1],
            train_epochs=flags_obj.train_epochs,
            compute_lr_on_cpu=True)
    else:
        raise NotImplementedError


    optimizer = common.get_optimizer(lr_schedule)

    if dtype == tf.float16:
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          optimizer, loss_scale)
    elif flags_obj.fp16_implementation == 'graph_rewrite':
      # `dtype` is still float32 in this case. We built the graph in float32 and
      # let the graph rewrite change parts of it float16.
      if not flags_obj.use_tf_function:
        raise ValueError('--fp16_implementation=graph_rewrite requires '
                         '--use_tf_function to be true')
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer, loss_scale)

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)

    if flags_obj.pretrained_filepath:
      logging.info('@@@load pretrained_filepath({})'.format(flags_obj.pretrained_filepath))
      status = checkpoint.restore(flags_obj.pretrained_filepath)
      status.assert_existing_objects_matched()
      x = model.output
      x = resnet_model.get_top_layer(x, dataset_conf.num_classes, use_l2_regularizer)
      model = models.Model(model.input, x, name='resnet50')
      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


    elif latest_checkpoint:
      checkpoint.restore(latest_checkpoint)
      current_step = optimizer.iterations.numpy()

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    categorical_cross_entopy_and_acc = losses.CategoricalCrossEntropyAndAcc(
                                          batch_size=flags_obj.batch_size,
                                          num_classes=dataset_conf.num_classes,
                                          label_smoothing=flags_obj.label_smoothing)
    trainable_variables = model.trainable_variables
    #logging.info('trainable variable = {}'.format(model.trainable_variables))

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        #labels = tf.squeeze(labels)
        #onehot_labels = tf.one_hot(labels, imagenet_preprocessing.NUM_CLASSES, dtype=tf.float32, axis=-1)
        #if mixup is not None:
        #  images, onehot_labels = mixup(images, onehot_labels)
        logits = model(images, training=True)
        logging.info('@@logits = {}'.format(logits))
        #loss = categorical_cross_entopy_and_acc.loss_and_update_acc_onehot(labels, onehot_labels, logits, training=True)
        loss = categorical_cross_entopy_and_acc.loss_and_update_acc(labels, logits, training=True)
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        if flags_obj.single_l2_loss_op:
          l2_loss = resnet_model.L2_WEIGHT_DECAY * 2 * tf.add_n([
              tf.nn.l2_loss(v)
              for v in trainable_variables
              if 'bn' not in v.name
          ])

          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(model.losses) / num_replicas)

        # Scale the loss
        if flags_obj.dtype == "fp16":
          loss = optimizer.get_scaled_loss(loss)

      grads = tape.gradient(loss, trainable_variables)

      # Unscale the grads
      if flags_obj.dtype == "fp16":
        grads = optimizer.get_unscaled_gradients(grads)

      optimizer.apply_gradients(zip(grads, trainable_variables))
      train_loss.update_state(loss)

    @tf.function
    def train_steps(iterator, steps):
      """Performs distributed training steps in a loop."""
      for _ in tf.range(steps):
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    def train_single_step(iterator):
      if strategy:
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))
      else:
        return step_fn(next(iterator))

    def test_step(iterator):
      """Evaluation StepFn."""
      def step_fn(inputs):
        images, labels = inputs
        logits = model(images, training=False)
        loss = categorical_cross_entopy_and_acc.loss_and_update_acc(labels, logits, training=False)
        #loss = tf.reduce_sum(loss) * (1.0/ flags_obj.batch_size)
        test_loss.update_state(loss)

      if strategy:
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))
      else:
        step_fn(next(iterator))

    if flags_obj.use_tf_function:
      train_single_step = tf.function(train_single_step)
      test_step = tf.function(test_step)

    if flags_obj.enable_tensorboard:
      summary_writer = tf.summary.create_file_writer(flags_obj.model_dir)
    else:
      summary_writer = None

    train_iter = iter(train_ds)
    time_callback.on_train_begin()
    for epoch in range(current_step // per_epoch_steps, train_epochs):
      train_loss.reset_states()
      categorical_cross_entopy_and_acc.training_accuracy.reset_states()

      steps_in_current_epoch = 0
      while steps_in_current_epoch < per_epoch_steps:
        time_callback.on_batch_begin(
            steps_in_current_epoch+epoch*per_epoch_steps)
        steps = _steps_to_run(steps_in_current_epoch, per_epoch_steps,
                              steps_per_loop)
        if steps == 1:
          train_single_step(train_iter)
        else:
          # Converts steps to a Tensor to avoid tf.function retracing.
          train_steps(train_iter, tf.convert_to_tensor(steps, dtype=tf.int32))
        time_callback.on_batch_end( steps_in_current_epoch+epoch*per_epoch_steps)
        steps_in_current_epoch += steps

      #temp_loss = array_ops.identity(categorical_cross_entopy_and_acc.training_loss).numpy()
      #temp_loss = categorical_cross_entopy_and_acc.training_loss.numpy()
      logging.info('Training loss: %s, accuracy: %s, cross_entropy: %s at epoch %d',
                   train_loss.result().numpy(),
                   categorical_cross_entopy_and_acc.training_accuracy.result().numpy(),
                   0.,
                   epoch + 1)
      #logging.info('@epoch{},,,, trainable variable = {}'.format(epoch +1, model.trainable_variables))

      if (not flags_obj.skip_eval and
          (epoch + 1) % flags_obj.epochs_between_evals == 0):
        test_loss.reset_states()
        categorical_cross_entopy_and_acc.test_accuracy.reset_states()

        test_iter = iter(test_ds)
        for _ in range(eval_steps):
          test_step(test_iter)

        logging.info('Test loss: %s, accuracy: %s%% at epoch: %d',
                     test_loss.result().numpy(),
                     categorical_cross_entopy_and_acc.test_accuracy.result().numpy(),
                     epoch + 1)

      if flags_obj.enable_checkpoint_and_export:
        checkpoint_name = checkpoint.save(
            os.path.join(flags_obj.model_dir,
                         'model.ckpt-{}'.format(epoch + 1)))
        logging.info('Saved checkpoint to %s', checkpoint_name)
        #Add keras save
        #export_path = os.path.join(flags_obj.model_dir, 'saved_model')
        #export_path = os.path.join(flags_obj.model_dir, 'h5')
        #model.save(export_path, include_optimizer=False)
        #model.save_weights(export_path, save_format='h5')

      if summary_writer:
        current_steps = steps_in_current_epoch + (epoch * per_epoch_steps)
        with summary_writer.as_default():
          #tf.summary.scalar('train_cross_entropy', categorical_cross_entopy_and_acc.training_loss.numpy(), current_steps)
          #tf.summary.image('Training data', train_iter[0][0], max_outputs=1, step=current_steps)
          tf.summary.scalar('train_loss', train_loss.result(), current_steps)
          tf.summary.scalar('train_accuracy', categorical_cross_entopy_and_acc.training_accuracy.result(),
                            current_steps)
          lr_for_monitor = lr_schedule(current_steps)
          if callable(lr_for_monitor):
            lr_for_monitor = lr_for_monitor()
          tf.summary.scalar('learning_rate', lr_for_monitor, current_steps)
          tf.summary.scalar('eval_loss', test_loss.result(), current_steps)
          tf.summary.scalar(
              'eval_accuracy', categorical_cross_entopy_and_acc.test_accuracy.result(), current_steps)

    time_callback.on_train_end()
    if summary_writer:
      summary_writer.close()

    eval_result = None
    train_result = None
    if not flags_obj.skip_eval:
      eval_result = [test_loss.result().numpy(),
                     categorical_cross_entopy_and_acc.test_accuracy.result().numpy()]
      train_result = [train_loss.result().numpy(),
                      categorical_cross_entopy_and_acc.training_accuracy.result().numpy()]

    stats = build_stats(train_result, eval_result, time_callback)
    return stats


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_keras_flags(pretrained_filepath=True)
  app.run(main)
