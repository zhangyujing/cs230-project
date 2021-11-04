import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
TRAIN_CSV = "./tsv/train.tsv"
DEV_CSV = "./tsv/dev.tsv"

# Load training set
train_df = pd.read_table("./tsv/train.tsv",header = None, names = ['id','qid1','qid2',
                                                              'question1', 'question2','is_duplicate'])
# Load dev set
dev_df = pd.read_table("./tsv/dev.tsv",header = None, names = ['id','qid1','qid2','question1', 'question2','is_duplicate'])
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]
    dev_df[q + '_n'] = dev_df[q]
    




# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
use_w2v = True

dev_df, embeddings = make_w2v_embeddings(dev_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)


X_train = train_df[['question1_n', 'question2_n']]
Y_train =  train_df['is_duplicate']
X_validation = dev_df[['question1_n', 'question2_n']]
Y_validation = dev_df['is_duplicate']

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# --

# Model variables
gpus = 1#2
batch_size = 1024 * gpus
n_epoch = 20
n_hidden = 50

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
# CNN
# x.add(Conv1D(250, kernel_size=5, activation='relu'))
# x.add(GlobalMaxPool1D())
# x.add(Dense(250, activation='relu'))
# x.add(Dropout(0.3))
# x.add(Dense(1, activation='sigmoid'))
# LSTM
x.add(LSTM(n_hidden))

shared_model = x

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

#if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
 #   model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
shared_model.summary()

# Start trainings
training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

# Added the logits layer followed by Dense to get binary_crossentropy loss
# becauase our targets are binary not continuous

# malstm_distance_logits = Dense(1)(malstm_distance)
# model = Model(inputs=[left_input, right_input], outputs=[malstm_distance_logits])

malstm_distance_logits = Dense(1)(malstm_distance)
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance_logits])


import datetime
model.save('./'  + str(datetime.datetime.now()) + 'SiameseLSTM.h5')

# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('./'  + str(datetime.datetime.now()) + 'history-graph.png')

print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
print("Done.")

