#!/usr/bin/env python
#
#                                   lstm.py
#

import tensorflow as tf
import numpy as np
import os
import random
import datetime

from keras.layers import Input, Dense, LSTM, ConvLSTM2D, Reshape, Embedding, Permute, RepeatVector, Multiply, Lambda
from keras import initializers
from keras import regularizers
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers import SimpleRNNCell, RNN
from keras.layers import LeakyReLU
from keras.layers.core import Activation
from keras.models import Sequential, Model
from keras import metrics
from tensorflow.keras import optimizers
import keras.backend as K
import tensorflow_addons as tfa
import keras_metrics
from keras.callbacks import ModelCheckpoint, CSVLogger
import json
import pandas as pd

"""
from keras.callbacks import Callback
from keras.activations import get as get_activation
from glob import glob
from sklearn.model_selection import train_test_split
"""


def randomize_data(x, y):

    length = x.shape[0]
    l = random.sample(range(length), length)
    data_x = [x[i] for i in l] 
    data_y = [y[i] for i in l] 
    return np.array(data_x), np.array(data_y)


def get_param(param, model):

    #global param
    param['optimizer'] = {}
    for tug in ['_hyper']:
        print (model.optimizer.__dict__[tug])
        param['optimizer'][tug] = {k:float(v) for k, v in model.optimizer.__dict__[tug].items()}
    param['optimizer']['_name'] = model.optimizer.__dict__['_name']


def get_optimal(result, fit_result):

    #global result
    result['score'] = {}
    i = np.argmax(fit_result.history['val_precision'])
    for attr in ['binary_accuracy', 'val_binary_accuracy',
        'precision', 'val_precision', 'loss', 'val_loss']:
        result['score'][attr] = float(fit_result.history[attr][i])


def plot_accuracy(i_file, o_file):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(o_file)
    df = pd.read_csv(i_file, index_col=False)
    plt.plot(df.epoch, df.binary_accuracy, 'k', label = 'Accuracy')
    plt.plot(df.epoch, df.val_binary_accuracy, 'k--', label = 'Val Accuracy')
    plt.plot(df.epoch, df.precision,    'r',    label = 'Precision')
    plt.plot(df.epoch, df.val_precision,'r--',  label = 'Val Precision')
    plt.plot(df.epoch, df.recall,       'b',    label = 'Recall')
    plt.plot(df.epoch, df.val_recall,   'b--',  label = 'Val Recall')
    plt.ylim((0,1.2))
    plt.legend()
    print ('OUTPUT PNG FILE', o_file)
    pp.savefig()
    plt.clf()

    plt.plot(df.epoch, df.loss, label = 'Loss')
    plt.plot(df.epoch, df.val_loss, label = 'Validation Loss')
    plt.legend()
    pp.savefig()
    pp.close()

def plot_separate(i_file, o_file):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lw = 1      # linewidth: 1 for thin line
    pp = PdfPages(o_file)
    df = pd.read_csv(i_file, index_col=False)
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.plot(df.epoch, df.binary_accuracy,      'k', label = 'Accuracy', linewidth=lw)
    ax1.plot(df.epoch, df.val_binary_accuracy,  'r', label = 'Val Accuracy', linewidth=lw)
    ax2.plot(df.epoch, df.precision,    'k', label = 'Precision', linewidth=lw)
    ax2.plot(df.epoch, df.val_precision,'r', label = 'Val Precision', linewidth=lw)
    ax3.plot(df.epoch, df.recall,       'k', label = 'Recall', linewidth=lw)
    ax3.plot(df.epoch, df.val_recall,   'r', label = 'Val Recall', linewidth=lw)
    ax1.set_ylim((0,1))
    ax2.set_ylim((0,1))
    ax3.set_ylim((0,1))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.tight_layout()
    print ('OUTPUT PNG FILE', o_file)
    pp.savefig()
    fig.clf()

    ### LOSS & VAL LOSS ###
    plt.plot(df.epoch, df.loss, label = 'Loss', linewidth=lw)
    plt.plot(df.epoch, df.val_loss, label = 'Validation Loss', linewidth=lw)
    plt.legend()
    pp.savefig()
    pp.close()


def count_label(y):

    ser_y = pd.Series(y)
    ser_y = ser_y.groupby(ser_y).count()
    mat = np.array([list(ser_y.index), list(ser_y.values)]).transpose()
    #mat = list([list(ser_y.index), list(ser_y.values)]).transpose()
    print (mat)
    return mat

IFILE = 'data/metamorphoses/npy'
x_file = IFILE + '/train_X.npy'
y_file = IFILE + '/train_Y.npy'

#lr = 0.000003
#lr = 0.00001
#lr = 0.00003
#lr = 0.0001
#lr = 0.0003
#lr = 0.001
#lr = 0.003
lr = 0.01

#decay = 0.001
#decay = 0.0003
#decay = 0.0001
decay = 0

relu_alpha = 0.01

#regu = 0.001
regu = 0.003
#regu = 0.01
#regu = 0.03

#embed_size = 4
#embed_size = 8
#embed_size = 16
#embed_size = 32
embed_size = 64
#embed_size = 128
#embed_size = 256

#rnn_units = 4
#rnn_units = 8
#rnn_units = 16
#rnn_units = 32
rnn_units = 64
#rnn_units = 128
#rnn_units = 256
#rnn_units = 516
#rnn_units = 1024

#n_epoch = 3         # FOR TEST
#n_epoch = 10
#n_epoch = 100
#n_epoch = 200
n_epoch = 400
#n_epoch = 1000
#n_epoch = 3000
#n_epoch = 10000

batch_size = 16
#batch_size = 128

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

param = {}
param['files'] = {}

x_train , y_train = np.load(x_file), np.load(y_file)

now = datetime.datetime.now()
timecode = now.strftime("%y%m%d-%H%M")     # like 210813-0909t-
history_dir = 'result/' + timecode
if not os.path.isdir(history_dir):
    os.mkdir(history_dir)

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.70)
print ('x_train', x_train.shape)
print ('y_train', y_train.shape)


samples, timesteps, features = x_train.shape

x_in = Input(shape=(timesteps, features))
#x_in = Input(shape=(None,))
x = Dense(embed_size)(x_in)
#x = Embedding(timesteps, embed_size)(x_in)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(rnn_units, return_sequences=True)(x)
x = Dropout(0.2)(x)
"""
x_in = Input(shape=(timesteps, features))
x = Dense(
    embed_size,
    kernel_initializer=initializers.he_normal(seed=None),
    bias_initializer=initializers.he_normal(seed=None),
    #kernel_regularizer=regularizers.l2(regu),
    #bias_regularizer=regularizers.l2(regu)
    )(x_in)

lnLSTMCell = tfa.rnn.LayerNormLSTMCell(rnn_units,
    activation  = 'sigmoid',
    kernel_regularizer=regularizers.l2(regu),
    bias_regularizer=regularizers.l2(regu),
    )
x = RNN(lnLSTMCell,
    return_sequences=True
    )(x)

lnLSTMCell2 = tfa.rnn.LayerNormLSTMCell(rnn_units,
    activation  = 'sigmoid',
    kernel_regularizer=regularizers.l2(regu),
    bias_regularizer=regularizers.l2(regu),
    )
x = RNN(lnLSTMCell2,
    return_sequences=True
    )(x)
"""

e = Dense(1,
        activation = LeakyReLU(alpha=relu_alpha),
        kernel_initializer=initializers.he_normal(seed=None)
        )(x)
e = Reshape([-1])(e)

alpha = Activation('softmax')(e)

c = Permute([2, 1])(RepeatVector(rnn_units)(alpha))
c = Multiply()([x, c])
c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)

out = Dense(timesteps, activation = 'sigmoid', name = 'foot')(c)

model = Model(x_in, out)
att_model = Model(x_in, alpha)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=lr, decay=decay),
    #optimizer=optimizers.Adam(lr=lr, decay=decay),
    #optimizer=optimizers.Adam(lr=lr, beta_1=0.001, decay=decay),
    #optimizer=optimizers.SGD(lr=lr, decay=decay),
    #optimizer=optimizers.Adagrad(lr=lr, decay=decay),
    metrics=[metrics.binary_accuracy, keras_metrics.precision(), keras_metrics.recall()]
)

att_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(),
    metrics=[metrics.binary_accuracy]
)
#model.summary()
#att_model.summary()

file_main   = '{}/{}'.format(history_dir, timecode)
json_file   = file_main + '-param.txt'
csv_file    = file_main + '-log.csv'
desc_file   = file_main + '-model.txt'
att_file    = file_main + '-att_model.txt'
model_file  = file_main + '-model.hdf5'
result_file = file_main + '-result.txt'
pdf_file    = file_main + '-loss.pdf'

print ('SAVED MODEL', desc_file)
with open(desc_file, 'w') as t_handle:
    model.summary(print_fn=lambda x: t_handle.write(x + '\n'))
with open(att_file, 'w') as a_handle:
    att_model.summary(print_fn=lambda x: a_handle.write(x + '\n'))

param['files']['score_log_file'] = csv_file
param['files']['param_file'] = json_file
param['regularization'] = regu
param['relu_alpha']     = relu_alpha
param['embedding_size'] = embed_size
param['rnn_units']      = rnn_units
param['epochs']         = n_epoch
param['batch_size']     = batch_size

get_param(param, model)
param_text = json.dumps(param, indent=2)
with open(json_file, 'w') as json_handle:
    json_handle.write(param_text)

callbacks = [
    ModelCheckpoint(filepath=history_dir + '/' + timecode +
        '-{val_binary_accuracy:.4f}-{epoch:04d}.hdf5',
        monitor='val_binary_accuracy', verbose=1, save_best_only=True),
    CSVLogger(csv_file, append=True)
]

fit_result = model.fit(x_train, y_train, epochs=n_epoch, batch_size=batch_size,
          validation_data=(x_train, y_train), callbacks=callbacks)

print ('SAVE MODEL', model_file)
model.save(model_file)
latest_model = 'latest-model.hdf5'
if os.path.isfile(latest_model):
    os.remove(latest_model)
print ('SAVE MODEL', latest_model)
os.symlink(model_file, latest_model)
#model.save(latest_model)

result = {}
result['files'] = result_file
get_optimal(result, fit_result)
print (json.dumps(result, indent=2))
result_text = json.dumps(result, indent=2)
with open(result_file, 'w') as result_handle:
    result_handle.write(result_text)

plot_separate(csv_file, pdf_file)

#from validate import validate
#validate(model_file, dir_val)
