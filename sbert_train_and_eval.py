import functools
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import dataset
import model
import utils
from official.common import distribute_utils
from official.nlp import optimization
from official.nlp.modeling import networks
from official.utils.misc import keras_utils

flags.DEFINE_enum('mode', 'train_and_eval', ['train_and_eval', 'predict'], 'One of "train_and_eval" and "predict".')
flags.DEFINE_string('model_dir', '', 'Model directory.')
flags.DEFINE_string('train_data_path', '', 'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', '', 'Path to evaluation data for BERT classifier.')
flags.DEFINE_integer('train_data_size', 384290, 'Number of training samples to use.')
flags.DEFINE_integer('eval_data_size', 10000, 'Number of evaluation samples to use.')
flags.DEFINE_integer('num_train_epochs', 50, 'Number of epochs for training.')
flags.DEFINE_integer('train_batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 16, 'Batch size for evaluation.')
flags.DEFINE_string('init_checkpoint', '', 'Checkpoint for initialization.')
flags.DEFINE_string('distribution_strategy', '', 'Distribution strategy.')
flags.DEFINE_integer('num_gpus', 0, 'Number of GPUs.')
flags.DEFINE_string('tpu', None, 'TPU address.')

# Model parameters
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_enum('pooling', 'mean', ['cls', 'mean', 'max', 'mean_max', 'strided_mean'], 'Pooling mechanism.')
flags.DEFINE_enum('compute_similarity', 'dense', ['cosine_similarity', 'dense', 'scaled_l1'], 'Similarity measures.')
flags.DEFINE_float('learning_rate', 3e-5, 'learning rate.')
flags.DEFINE_integer('vocab_size', 30522, 'Vocalubary size.')

flags.DEFINE_float('dynamic_masking_rate', 0.0, 'If > 0, the SSL auxilary task is enabled with dynamic masking.')

# Eval parameters
flags.DEFINE_float('question_word_penalty_weight', 0.0, 'Question word penalty weight during evaluation.')
flags.DEFINE_float('pred_threshold', 0.5, 'Threshold for similarity prediction.')

FLAGS = flags.FLAGS


def get_loss_fn():
  """Gets the binary classification loss function."""

  def classification_loss_fn(labels, logits):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.1, reduction=tf.keras.losses.Reduction.SUM)
    return binary_cross_entropy(labels, logits)

  return classification_loss_fn


def get_dataset_fn(input_file,
                   max_seq_length,
                   global_batch_size,
                   is_training):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    ds = dataset.create_dataset(
        input_file,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return ds

  return _dataset_fn


def run_classifier(strategy, model_dir, epochs, steps_per_epoch, eval_steps,
                   warmup_steps, init_checkpoint, train_input_fn,
                   eval_input_fn):
  """Train the classifier."""
  max_seq_length = FLAGS.max_seq_length
  pooling = FLAGS.pooling
  compute_similarity = FLAGS.compute_similarity

  loss_fn = (get_loss_fn())
  metric_fn = [
      functools.partial(
          tf.keras.metrics.BinaryCrossentropy, 'loss', dtype=tf.float32),
      functools.partial(
          tf.keras.metrics.BinaryAccuracy, 'accuracy', dtype=tf.float32),
      functools.partial(
          tf.keras.metrics.Precision, name='precision', dtype=tf.float32),
      functools.partial(
          tf.keras.metrics.Recall, name='recall', dtype=tf.float32),
  ]

  # Start training using Keras compile/fit API.
  with strategy.scope():
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn else None
    enc = networks.BertEncoder(
        vocab_size=FLAGS.vocab_size, num_layers=12, type_vocab_size=2)
    base_classifier = model.Classifier(enc, max_seq_length, pooling=pooling,
                                       compute_similarity=compute_similarity)
    if FLAGS.dynamic_masking_rate > 0:
      classifier = model.SSLClassifier(base_classifier, max_seq_length,
                                       FLAGS.dynamic_masking_rate)
    else:
      classifer = base_classifier
    classifier.optimizer = optimization.create_optimizer(
        FLAGS.learning_rate, steps_per_epoch * epochs, warmup_steps)
    optimizer = classifier.optimizer

    if init_checkpoint:
      # load pretrained encoder
      bert_checkpoint = tf.train.Checkpoint(model=enc)
      bert_checkpoint.restore(
          init_checkpoint).assert_existing_objects_matched().run_restore_ops()
      logging.info('load ckpt done')
 
    if FLAGS.dynamic_masking_rate > 0:
      loss_weights = {}
      loss_weights['main_p'] = 1.0
      loss_weights['ssl_p'] = 0.1
      losses = {}
      losses['main_p'] = loss_fn
      losses['ssl_p'] = loss_fn
      metrics = [fn() for fn in metric_fn]
    else:
      loss_weights = None
      losses = loss_fn
      metrics = {'main_p': [fn() for fn in metric_fn] }     

    classifier.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
        steps_per_execution=100)

    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint = tf.train.Checkpoint(model=classifier, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=None,
        step_counter=optimizer.iterations,
        checkpoint_interval=5000)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    callbacks = [summary_callback, checkpoint_callback]

    history = classifier.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=callbacks)
    stats = {'total_training_steps': steps_per_epoch * epochs}
    if 'loss' in history.history:
      stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
      stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return classifier, stats


def train_and_eval(strategy,
                   train_input_fn=None,
                   eval_input_fn=None,
                   init_checkpoint=None):
  """Run training and evaluation."""
  epochs = FLAGS.num_train_epochs
  train_data_size = FLAGS.train_data_size

  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
  eval_steps = int(math.ceil(FLAGS.eval_data_size / FLAGS.eval_batch_size))

  trained_model, _ = run_classifier(strategy, FLAGS.model_dir, epochs,
                                    steps_per_epoch, eval_steps, warmup_steps,
                                    init_checkpoint, train_input_fn,
                                    eval_input_fn)

  return trained_model


def get_predictions_and_labels(strategy, trained_model, eval_input_fn):
  """Obtains predictions of trained model on test data."""

  @tf.function
  def predict_step(iterator):
    """Computes predictions on distributed devices."""

    def _predict_step_fn(inputs):
      """Replicated predictions."""
      inputs, labels = inputs
      probabilities = trained_model(inputs, training=False)
      if FLAGS.dynamic_masking_rate > 0:
        probabilities = probabilities['main_p']
      return probabilities, labels

    outputs, labels = strategy.run(_predict_step_fn, args=(next(iterator),))
    outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                    outputs)
    labels = tf.nest.map_structure(strategy.experimental_local_results, labels)
    return outputs, labels

  test_iter = iter(strategy.distribute_datasets_from_function(eval_input_fn))
  all_predictions, all_labels = list(), list()
  try:
    with tf.experimental.async_scope():
      while True:
        probs, labels = predict_step(test_iter)
        for prob, label in zip(probs, labels):
          all_predictions.extend(prob.numpy())
          all_labels.extend(label.numpy())
  except (StopIteration, tf.errors.OutOfRangeError):
    tf.experimental.async_clear_error()

  return all_predictions, all_labels


def run():
  """Run the classification."""
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  eval_input_fn = get_dataset_fn(
      FLAGS.eval_data_path,
      FLAGS.max_seq_length,
      FLAGS.eval_batch_size,
      is_training=False)
  init_checkpoint = FLAGS.init_checkpoint

  if FLAGS.mode == 'predict':
    with strategy.scope():
      enc = networks.BertEncoder(
          vocab_size=FLAGS.vocab_size, num_layers=12, type_vocab_size=2)
      base_classifier = model.Classifier(enc, FLAGS.max_seq_length, pooling=FLAGS.pooling,
                                         compute_similarity=FLAGS.compute_similarity,
                                         question_word_penalty_weight=FLAGS.question_word_penalty_weight,
                                         pred_threshold=FLAGS.pred_threshold)
      if FLAGS.dynamic_masking_rate > 0:
        classifier = model.SSLClassifier(base_classifier, max_seq_length,
                                         FLAGS.dynamic_masking_rate)
      else:
        classifer = base_classifier
      checkpoint = tf.train.Checkpoint(model=classifier)
      latest_checkpoint_file = (
          init_checkpoint or tf.train.latest_checkpoint(FLAGS.model_dir))
      assert latest_checkpoint_file
      logging.info('Checkpoint file %s found and restoring from '
                   'checkpoint', latest_checkpoint_file)
      checkpoint.restore(
          latest_checkpoint_file).assert_existing_objects_matched()
      preds, labels = get_predictions_and_labels(strategy, classifier, eval_input_fn)
    utils.write_prediction_outputs(preds, labels)
    return

  train_input_fn = get_dataset_fn(
      FLAGS.train_data_path,
      FLAGS.max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)
  train_and_eval(strategy, train_input_fn, eval_input_fn, init_checkpoint)


def main(_):
  run()


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  app.run(main)
