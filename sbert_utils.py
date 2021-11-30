"""Util functions."""
import os

import tensorflow as tf

def write_prediction_outputs(preds, labels):
  """Write the prediction outputs and compute the metrics."""
  output_predict_file = os.path.join(FLAGS.model_dir, 'test_results.tsv')
  with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
    logging.info('***** Predict results *****')
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for prob, label in zip(preds, labels):
      pred = int(prob > 0.5)
      if pred:
        if label:
          true_pos = true_pos + 1
        else:
          false_pos = false_pos + 1
      else:
        if label:
          false_neg = false_neg + 1
        else:
          true_neg = true_neg + 1
      output_line = str(prob) + ': ' + str(pred) + ' | ' + str(label) + '\n'
      writer.write(output_line)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    writer.write('accuracy: ' +
                 str((true_pos + true_neg) /
                     (true_pos + false_pos + true_neg + false_neg)) + '\n')
    writer.write('precision: ' + str(precision) + '\n')
    writer.write('recall: ' + str(recall) + '\n')
    writer.write('f1: ' + str(2 * precision * recall / (precision + recall)) + '\n')
