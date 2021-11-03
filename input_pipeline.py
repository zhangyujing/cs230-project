import tensorflow as tf

def create_dataset(input_file,
                   seq_length,
                   batch_size,
                   is_training=True,
                   input_pipeline_context=None):
  """Creates a TF dataset."""
  dataset = tf.data.TFRecordDataset(input_file)

  features = {
      'left_input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'left_input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'left_segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'right_input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'right_input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'right_segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label': tf.io.FixedLenFeature([], tf.float32),
  }
  dataset = dataset.map(
      lambda record: tf.io.parse_single_example(record, features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  def _select_data_from_record(record):
    x = {
        'left_input_ids': record['left_input_ids'],
        'left_input_mask': record['left_input_mask'],
        'left_input_type_ids': record['left_segment_ids'],
        'right_input_ids': record['right_input_ids'],
        'right_input_mask': record['right_input_mask'],
        'right_input_type_ids': record['right_segment_ids']
    }
    y = record['label']
    return (x, y)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
