#!/usr/bin/env python
#
#                           validate.py
#

from make_data import Counter
import tensorflow_addons as tfa
from tensorflow_addons.rnn import LayerNormLSTMCell
import numpy as np

def show_predict(X, Y, m_file, counter):

    import tensorflow as tf
    import keras
    import keras_metrics

    ### READ MODEL ###
    model = tf.keras.models.load_model(
        m_file,
        custom_objects={
            'LeakyReLU': keras.layers.LeakyReLU,
            'binary_precision': keras_metrics.precision(),
            'binary_recall':    keras_metrics.recall()
        })

    pred = model.predict(X)
    for x, y, p in zip(X, Y, pred):
        s = ''.join([counter.convert(v) for v in x])
        print (s.replace('#', ' '))     # ALPHABET
        print (''.join(['/' if (c == 1) else ' ' for c in y]))
        print (get_predict(p))
        print (get_prob(p))
        print ()


def get_prob(p):

    z = ''.join(np.floor(p * 10).astype(int).astype(str))
    z = z.replace('0', ' ')
    return ''.join(z)


def get_predict(p):

    z = (p >= 0.5)#.astype(str)
    z = ['/' if i else ' ' for i in z]
    return ''.join(z)


if __name__ == '__main__':

    import numpy as np
    import pickle
    #tc = '211124-1706'
    #tc = '211124-1820'
    tc = '211124-1836'
    DFILE = 'data/metamorphoses/npy'
    CFILE = 'data/metamorphoses/counter.pkl'
    #X = np.load(DFILE + '/train_X.npy')
    #Y = np.load(DFILE + '/train_Y.npy')
    X = np.load(DFILE + '/test_X.npy')
    Y = np.load(DFILE + '/test_Y.npy')
    #MFILE = 'result/%s/%s-model.hdf5' % (tc, tc)
    MFILE = './latest-model.hdf5'
    #print ('MODEL', MFILE)
    #with open(MFILE, 'rb') as m_handle:
    #    model = pickle.load(m_handle)
    print ('COUNTER', CFILE)
    with open(CFILE, 'rb') as c_handle:
        counter = pickle.load(c_handle)
    show_predict(X, Y, MFILE, counter)
