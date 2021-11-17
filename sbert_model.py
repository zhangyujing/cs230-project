import collections

import tensorflow as tf


class Classifier(tf.keras.Model):
  """Sequence similarity classifier.

    Args:
      encoder: a tf.keras.Model defines an encoder network.
      max_seq_length: maximum sequence length.
      poolining: poolining mechanism. One of [cls, mean, max].
      compute_similarity: how to compute the similarity probability outputs.
        One of [cosine_similarity, dense].
  """

  def __init__(self,
               encoder,
               max_seq_length,
               dropout_rate=0.1,
               pooling='mean',
               compute_similarity='dense',
               **kwargs):
    self.max_seq_length = max_seq_length
    self.pooling = pooling
    self.compute_similarity = compute_similarity

    # Prepare inputs
    left_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='left_word_ids')
    left_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='left_mask')
    left_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='left_type_ids')
    left_inputs = [left_word_ids, left_mask, left_type_ids]

    right_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='right_word_ids')
    right_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='right_mask')
    right_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='right_type_ids')

    right_inputs = [right_word_ids, right_mask, right_type_ids]
    inputs = {
        'left_input_ids': left_word_ids,
        'left_input_mask': left_mask,
        'left_input_type_ids': left_type_ids,
        'right_input_ids': right_word_ids,
        'right_input_mask': right_mask,
        'right_input_type_ids': right_type_ids
    }

    expanded_left_mask = tf.expand_dims(
        tf.cast(left_mask, dtype=tf.float32), axis=2)
    expanded_right_mask = tf.expand_dims(
        tf.cast(right_mask, dtype=tf.float32), axis=2)
    # Get sequence embeddings
    left_sequence_output, left_cls_output = encoder(left_inputs)
    right_sequence_output, right_cls_output = encoder(right_inputs)
    left_sequence_output = left_sequence_output * expanded_left_mask
    right_sequence_output = right_sequence_output * expanded_right_mask

    # Pooling
    if pooling == 'cls':
      # Outputs on the first token 'CLs'
      left_outputs = left_cls_output
      right_outputs = right_cls_output
    elif pooling == 'mean':
      left_outputs = tf.reduce_sum(
          left_sequence_output, axis=1) / tf.reduce_sum(expanded_left_mask, axis=1)
      right_outputs = tf.reduce_sum(
          right_sequence_output, axis=1) / tf.reduce_sum(expanded_right_mask, axis=1)
    elif pooling == 'max':
      left_outputs = tf.reduce_max(left_sequence_output, axis=1)
      right_outputs = tf.reduce_max(right_sequence_output, axis=1)
    else:
      raise ValueError('Pooling %s is not supported: %s' % pooling)

    # Compute similarity
    if compute_similarity == 'cosine_similarity':
      cos_similarity = tf.reduce_sum(
          tf.nn.l2_normalize(left_outputs, axis=1) *
          tf.nn.l2_normalize(right_outputs, axis=1),
          axis=1)
      prob = (cos_similarity + 1) / 2
    elif compute_similarity == 'scaled_l1':
      prob = tf.exp(-tf.reduce_sum(tf.abs(
        tf.nn.l2_normalize(left_outputs, axis=1) - tf.nn.l2_normalize(right_outputs, axis=1)),
                                   axis=1))
    elif compute_similarity == 'dense':
      concat_outputs = tf.concat(
          [left_outputs, right_outputs,
           tf.abs(left_outputs - right_outputs)],
          axis=1)
      outputs = tf.keras.layers.Dense(64, activation='relu')(concat_outputs)
      prob = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
    else:
      raise ValueError('compute_similarity %s is not supported: %s' %
                       compute_similarity)

    super(Classifier, self).__init__(inputs=inputs, outputs=prob, **kwargs)
    self._encoder = encoder

    config_dict = {
        'encoder': self._encoder,
        'max_seq_length': self.max_seq_length,
        'pooling': self.pooling,
        'compute_similarity': self.compute_similarity
    }
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)

  @property
  def checkpoint_items(self):
    items = dict(encoder=self._encoder)
    return items

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)