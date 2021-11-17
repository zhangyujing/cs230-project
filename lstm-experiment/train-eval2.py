# reference: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
import pandas as pd
import matplotlib

from keras import backend as K


import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding,LSTM, Dense

from sklearn.model_selection import train_test_split

from util import make_embeddings, padding, ManDist


# helper function for calculating score
# reference: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# Load training set
train = pd.read_table("./tsv/train.tsv",header = None, names = ['id','qid1','qid2',
                                                              'question1', 'question2','is_duplicate'])
X = train[['question1', 'question2']]
Y = train['is_duplicate'].values

# embeddings parameters
embedding_dim = 300
seqLength = 30
train, embeddings = make_embeddings(train, embedding_dim=embedding_dim)

# Set up train validation

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=size)

X_train = padding(X_train, seqLength)
X_val = padding(X_val, seqLength)

# Define the shared model
# Model variables
batch_size = 1024
n_hidden = 150

# reference: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
#shared layer
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(seqLength,), trainable=False))

x.add(LSTM(n_hidden))

shared = x
# two inputs
left = Input(shape=(seqLength,), dtype='int32')
right = Input(shape=(seqLength,), dtype='int32')

#  Manhattan Distance layer
left_in, right_in = shared(left), shared(right)
malstm_distance = ManDist()([left_in, right_in ])
#define model
model = Model(inputs=[left, right], outputs=[malstm_distance])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(),  metrics=['acc',f1_m,precision_m, recall_m])


epoch = 20 #20
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=epoch,
                           validation_data=([X_val['left'], X_val['right']], Y_val))


model.save('./savedModel.h5')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./history.png')

print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")


prediction = model.predict([X_test['left'], X_test['right']])
loss, accuracy, f1_score, precision, recall = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=0)
print(prediction)
print(accuracy)
print(f1_score)
print(precision)
print(recall)