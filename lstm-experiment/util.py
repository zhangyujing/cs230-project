
import re
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

import gensim
from gensim import models
from gensim.models import KeyedVectors


import itertools

import nltk
nltk.download('stopwords')


#text cleaning: reference: https://www.kaggle.com/currie32/the-importance-of-cleaning-text
def clean(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


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



def make_embeddings(df, embedding_dim):
    dict = {}
    dict_cnt = 0

    dict_not_w2v = {}
    dict_not_w2v_cnt = 0

    # import word2ve
    word2vec = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin.gz", binary=True)
    # set up Stopwords
    stop = set(stopwords.words('english'))
    for index, row in df.iterrows():
        # Iterate through the text of both questions of the row
        for question in ['question1', 'question2']:

            res = []  # res -> question numbers representation
            for word in clean(row[question]):
                if word in stop:
                    continue
                # new word in  word2vec model.
                if word not in word2vec.vocab:
                    if word not in dict_not_w2v:
                        dict_not_w2v_cnt += 1
                        dict_not_w2v[word] = 1
                # new word, append it to dictionary.
                if word not in dict:
                    dict_cnt += 1
                    dict[word] = dict_cnt
                    res.append(dict_cnt)
                else:
                    res.append(dict[word])

            # Append question as number representation
            df.at[index, question + '_n'] = res
    #make embedding
    embeddings = np.random.randn(len(dict) + 1, embedding_dim) * 1 # This will be the embedding matrix
    embeddings[0] = 0  

    # Build the embedding matrix
    for word, index in dict.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)

    return df, embeddings


def padding(df, max_seq_length):
    # Split to dicts
    X = {'left': df['question1_n'], 'right': df['question2_n']}

    # Zero padding
    for dataset, s in itertools.product([X], ['left', 'right']):
        dataset[s] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset

#  reference: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
class ManDist(Layer):
    def __init__(self, **kwargs):
        self.res = None
        super(ManDist, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.res)

    #Manhattance distance beween 0 and 1
    def call(self, x, **kwargs):
        self.res = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.res

