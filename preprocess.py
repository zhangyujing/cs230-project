"""Convert an input tsv file to a tf_record file."""

import collections
import csv
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.nlp.bert import tokenization

flags.DEFINE_string("input_file", "", "Path of input tsv file.")
flags.DEFINE_string("output_file", "",  "Path of output file.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_string("vocab_file", "", "Vocabulary file for tokenization.")

FLAGS = flags.FLAGS

def sentence_to_ids(sentence, tokenizer, max_seq_length):
  # Consider [CLS] and [SEP]
  tokenized = tokenizer.tokenize(sentence)
  tokens = ["[CLS]"]
  tokens.extend(tokenized[0:(max_seq_length - 2)])
  tokens.append("[SEP]")

  ids = tokenizer.convert_tokens_to_ids(tokens)
  mask = [1] * len(ids)
  seg_id = 0
  segment_ids = [seg_id] * max_seq_length
  while len(ids) < max_seq_length:
    ids.append(0)
    mask.append(0)

  assert len(ids) == max_seq_length
  assert len(mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  def create_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  return create_feature(ids), create_feature(mask), create_feature(segment_ids)


def convert_single_example(features, tokenizer, max_seq_length):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  assert len(features) == 6
  idx = int(features[0])
  sentence0 = features[3]
  sentence1 = features[4]
  label = float(features[5])

  features = collections.OrderedDict()
  features["id"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[idx]))
  (features["left_input_ids"], features["left_input_mask"],
   features["left_segment_ids"]) = sentence_to_ids(sentence0, tokenizer,
                                                   max_seq_length)
  (features["right_input_ids"], features["right_input_mask"],
   features["right_segment_ids"]) = sentence_to_ids(sentence1, tokenizer,
                                                    max_seq_length)
  features["label"] = tf.train.Feature(
      float_list=tf.train.FloatList(value=[label]))

  return tf.train.Example(features=tf.train.Features(
      feature=features)), label == 1.0


def run():
  output_file = FLAGS.output_file
  if tf.io.gfile.exists(output_file):
    tf.io.gfile.remove(output_file)

  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  example_idx = 0
  logging.info("Start.")
  pos_num = 0
  # WordPiece tokenization
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
  with tf.io.gfile.GFile(FLAGS.input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    examples = []
    for features in reader:
      tf_example, pos = convert_single_example(features, tokenizer,
                                               FLAGS.max_seq_length)
      if pos:
        pos_num = pos_num + 1
      writer.write(tf_example.SerializeToString())

      example_idx = example_idx + 1
      if example_idx % 10000 == 0:
        logging.info("Finished writing %d examples", example_idx)
    logging.info("Finished writing %d examples", example_idx)
    logging.info("Positive %d examples", pos_num)

  writer.close()


def main(_):
  run()


if __name__ == "__main__":
  app.run(main)
