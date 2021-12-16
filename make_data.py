#!/usr/bin/env python
#   
#                               make_data.py
#
# ヘクサメトロスの区切り情報からfoot情報をつくる

import numpy as np

CHAR_LENGTH = 80

class Counter:

    def __init__(self):
        
        self.dict = {}

    def feed(self, s):

        for c in s:
            if c in self.dict:
                self.dict[c] += 1
            else:
                self.dict[c] = 1

    def make_keys(self):

        self.keys = sorted(self.dict.keys())
        self.c2i = {c:i for i, c in enumerate(self.keys)}
        print (self.c2i)
        return self.keys

    def convert(self, vec):

        return (''.join([self.keys[i] for i in range(len(self.keys)) if vec[i] == 1]))



def parse_hex(i_file, c_file):

    import numpy as np
    import pickle
    counter = Counter()
    w = []
    with open (i_file, 'r') as i_handle:
        for i, line in enumerate(i_handle):
            line = line.lower().rstrip()
            foot_list = line.split('/')
            w.append(foot_list)
            counter.feed(''.join(foot_list))

    counter.feed('#')       # add the terminal character
    counter.make_keys()
    #print ('COUNTER FILE', c_file)
    with open(c_file, 'wb') as c_handle:
        pickle.dump(counter, c_handle)

    key_len = len(counter.keys)
    z = []
    y_list = []
    for i, foot_list in enumerate(w):
        line = ''.join(foot_list)
        line = line + ('#' * (CHAR_LENGTH - len(line)))
        print (line)

        x_list = []
        for c in line:
            x = np.zeros(key_len, dtype=int)
            x[counter.c2i[c]] = 1
            x_list.append(x)
        #print (np.array(x_list))
        z.append(np.array(x_list))

        y = [0 for i in range(CHAR_LENGTH)]
        j = 0
        for f in foot_list:
            j += len(f)
            y[j-1] = 1
        print (''.join(['%s' % b for b in y]))
        y_list.append(y)

        if i >= HEAD - 1:
            break
    return np.array(z), np.array(y_list)


###
if __name__ == '__main__':

    import pickle
    from sklearn.model_selection import train_test_split

    IFILE = 'data/metamorphoses/metamorphoses_hexameter.txt'
    OFILE = 'data/metamorphoses/npy'
    CFILE = 'data/metamorphoses/counter.pkl'
    HEAD = 239

    
    x, y = parse_hex(IFILE, CFILE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    print ('x_train', x_train.shape, 'y_train', y_train.shape)
    #x_file = OFILE + '/train_X.npy'
    #y_file = OFILE + '/train_Y.npy'
    #print ('TRAIN X', x_file, x.shape)
    np.save(OFILE + '/train_X.npy', x_train)
    np.save(OFILE + '/train_Y.npy', y_train)
    #np.save(x_file, x)
    #print ('TRAIN Y', y_file, y.shape)
    print ('x_test', x_test.shape, 'y_test', y_test.shape)
    np.save(OFILE + '/test_X.npy', x_test)
    np.save(OFILE + '/test_Y.npy', y_test)
