import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, MaxPooling1D, Conv1D
from keras.layers import LSTM, Dropout, Embedding, GRU, Bidirectional
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD  # note add tensorflow
from sklearn.metrics import accuracy_score, roc_auc_score  # for evaluating classifier
import tensorflow as tf  # for tensorflow metrics
import keras_tuner as kt  # for tuning learning rates, and hidden layers

# model params
MAX_SEQUENCE_LENGTH = 400  # truncate posts at length 400
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 2048  # make sure each batch has a chance of obtaining positive examples
EPOCHS = 35  # change epochs
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]  # list of metrics

# visualize results


def plot_cm(labels, predictions, p=0.5):
    cm = tf.math.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total Positive: ', np.sum(cm[1]))


# set seed
np.random.seed(42)

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

# imbalanced data correction
# examine imbalance
neg, pos = np.bincount(df['popular'])
total = neg + pos
INITIAL_BIAS = np.log([pos/neg])
# class weights
# scaling by total/2 helps keep the loss to a similar magnitude.
# the sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
CLASS_WEIGHT = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

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
print('Loading in posts')
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

# hypermodel


def model_builder(hp):
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    # tune the number of hidden units
    hp_units = hp.Int('units', min_value=8, max_value=128, step=32)

    x = Bidirectional(LSTM(units=hp_units, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(len(possible_labels), activation='sigmoid')(x)

    model = Model(input_, output)
    # tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate),
        metrics=METRICS,
    )
    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hyper',
                     project_name='tuning')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(data, targets,
             epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
             callbacks=[stop_early],
             batch_size=BATCH_SIZE,
             class_weight=CLASS_WEIGHT
             )
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(data, targets,
                    epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
                    batch_size=BATCH_SIZE,
                    class_weight=CLASS_WEIGHT
                    )
val_loss_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#
hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
model_tuned = hypermodel.fit(data, targets, epochs=best_epoch, validation_split=VALIDATION_SPLIT,
                             batch_size=BATCH_SIZE,
                             class_weight=CLASS_WEIGHT
                             )

print('Building model...')
# original: train a lstm network with a single lstm
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
# x = LSTM(15, return_sequences=True)(x)  # \
x = Bidirectional(LSTM(15, return_sequences=True))(x)
# x = GRU(15, return_sequences = True)(x)
x = GlobalMaxPooling1D()(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model_lstm1 = Model(input_, output)
model_lstm1.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=METRICS  # keep track of list of metrics
)

# set callbacks to prevent overfit and save as checkpts
checkpoint_path = './checkpoint/cp.cpkt'
checkpoint_dir = os.path.dirname(checkpoint_path)
CALL_BACKS = [
    # how many epochs without improvement you allow before the cb interferes
    EarlyStopping(monitor='val_loss', patience=5,
                  mode='min', min_delta=0.0001),
    ModelCheckpoint(checkpoint_path, monitor='val_loss',
                    save_best_only=True, mode='min'
                    ),
]
# train model
print('Training model...')
r = model_lstm1.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    class_weight=CLASS_WEIGHT,
    callbacks=[CALL_BACKS],
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# roc auc score
plt.plot(r.history['auc'], label='auc')
plt.plot(r.history['val_auc'], label='val_auc')
plt.legend()
plt.show()

# recall
plt.plot(r.history['recall'], label='recall')
plt.plot(r.history['val_recall'], label='val_recall')
plt.legend()
plt.show()

# eval model
p = model_lstm1.predict(data)
plot_cm(targets[:, 1], p[:, 1])  # true vs predicted popular classifier

# roc auc score
# 0.944 for lstm, 0.956 for bidirectional lstm
auc = roc_auc_score(targets[:, 1], p[:, 1])
print(auc)

# save model
model_lstm1.save('./model/blind_lstm_v1.h5')
