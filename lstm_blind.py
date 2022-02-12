import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional, Dropout, Embedding
from keras.models import Model
from keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam, SGD  # note add tensorflow
from sklearn.metrics import roc_auc_score  # for evaluating classifier
import tensorflow as tf  # for tensorflow metrics

# model params
MAX_SEQUENCE_LENGTH = 400  # truncate posts at length 400
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

# load data
file_path = './datafiles/blindPosts/'  # original csvs separated by company
files = [file_path + f for f in os.listdir(file_path)]
df = pd.concat([pd.read_csv(f, parse_dates=True).iloc[:, 1:]
                for f in files])  # concat all csvs
# prepare data
df = df.rename({'post_like': 'likes', 'post_comment': 'comments'}, axis=1)
df['post_firm'] = df['post_firm'].map(lambda x: str(
    x).replace('/company/', '').rstrip('/'))  # op's employer
df['likes'] = df['likes'].map(lambda x: str(x).replace(',', ''))  # of likes
df['comments'] = df['comments'].map(
    lambda x: str(x).replace(',', ''))  # of comments

# change data types
df['likes'] = pd.to_numeric(df['likes'])
df['comments'] = pd.to_numeric(df['comments'])
df['post_timestamp'] = pd.to_datetime(df['post_timestamp'])
df = df.reset_index().set_index(
    ['company', 'post_timestamp']).sort_index().reset_index()

# finalize sample, give labels
df[~df['post_text'].duplicated()]
df['popular'] = np.where(df['likes'] > df['likes'].quantile(0.99), 1, 0)
df['controversial'] = np.where(
    df['comments'] > df['comments'].quantile(0.99), 1, 0)
# keep poster company if available
df = df[['post_text', 'post_firm', 'popular', 'controversial']]


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./datafiles/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# prepare text samples and their labels
print('Loading in comments')
train = df  # we use the entire data
sentences = train['post_text'].fillna('DUMMY_VALUE').values
possible_labels = ['controversial', 'popular']
targets = train[possible_labels].values  # N * 2 matrix

s = sorted(len(s) for s in sentences)
print("median sequence length: ", s[len(s)//2])
print("max sequence length: ", s[-1])
print("min sequence length: ", s[0])
# convert the sentences (strings) to ints
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# get word -> int mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
# pad sequences so that we get N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that trainable = False as to keep the embeddings fixed
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False,  # keep the embeddings fixed
)
print('Building model...')

# train a lstm network with a single lstm
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=[AUC(name='auc')],
)

print('Training model...')
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)

aucs = []
for j in range(2):  # two targets
    auc = roc_auc_score(targets[:, j], p[:, j])
    aucs.append(auc)

print(np.mean(aucs))
# check confusion matrix for imbalanced classification
con_mat = tf.math.confusion_matrix(
    labels=targets[:, 0], predictions=p[:, 0]).numpy()

# save model
model.save('./model/blind_lstm')
