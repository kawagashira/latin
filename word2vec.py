#!/usr/bin/env python
#
#                               word2vec.py
#

import xml.etree.ElementTree as ET


def parse_xml(i_file):

    tree = ET.parse(i_file)
    root = tree.getroot()
    body = root[0][0]
    div = root[0][0][0]
    w = [child.text for child in div if child.tag == 'l']
    return w


def normalize(text):

    w = []
    for line in text:
        s = ''.join([c for c in line.lower() if c not in STOPWORD])
        print (s)
        w.append(' '.join(s))
    return ' '.join(w)


def bigram(text):

    w = []
    for line in text:
        s = ''.join([c for c in line.lower() if c not in STOPWORD])
        print (s)
        phrase = []
        for word in s.split(' '):
            print ('word', word)
            for i in range(len(word) - 1):
                b = word[i:(i+2)]
                print (b, i, i+2)
                w.append(b)
                phrase.append(b)
        w.append(phrase)
    return w
    #return ' '.join(w)


def make_w2v(text):

    from gensim.models import word2vec
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #model = word2vec.Word2Vec(text, vector_size=100, min_count=10, window=10, alpha=0.001, epochs=4000)
    model = word2vec.Word2Vec(text, vector_size=50, min_count=10, window=3, alpha=0.001, epochs=500)
    print ('index', model.wv.index_to_key)
    return model


"""
import pandas as pd
import numpy as np
"""

def scatter_plot(w2v, OFILE):

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    #dat = w2v.wv.vectors.transpose()
    dat = w2v.wv.vectors
    print ('dat w2v vectors', dat.shape) 
    #model = TruncatedSVD()
    model = TSNE()
    model.fit(dat)
    #X = model.components_.transpose()
    X = model.embedding_
    print ('X', X.shape)
    plt.scatter(X[:, 0], X[:, 1], marker=' ')
    for p, c in zip(X, w2v.wv.index_to_key):
        plt.text(p[0], p[1], c, ha='center', va='center')
    plt.show()
    plt.savefig(OFILE)
    plt.close()


if __name__ == '__main__':

    STOPWORD = '"\'\:\;\“\”\.\,\(\)\!\?'
    IFILE = 'data/metamorphoses/book1.xml'
    OFILE = 'result/metamorphoses-svd.png'

    text = parse_xml(IFILE)
    #spl_text = normalize(text)
    spl_text = bigram(text)
    print (spl_text)
    model = make_w2v(spl_text)
    scatter_plot(model, OFILE)
