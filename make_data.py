#!/usr/bin/env python
#   
#                               parse_hexameter.py
#
# ヘクサメトロスの区切り情報からfoot情報をつくる

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



def parse_hexameter(i_file):

    import numpy as np
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
    key_len = len(counter.keys)
    z = []
    for i, foot_list in enumerate(w):
        line = ''.join(foot_list)
        line = line + ('#' * (CHAR_LENGTH - len(line)))
        print (line)

        x_list = []
        for c in line:
            x = np.zeros(key_len, dtype=int)
            x[counter.c2i[c]] = 1
            x_list.append(x)
        print (np.array(x_list))
        z.append(np.array(x_list))

        y = [0 for i in range(CHAR_LENGTH)]
        j = 0
        for f in foot_list:
            j += len(f)
            y[j-1] = 1
        print (''.join(['%s' % b for b in y]))


        if i >= 10:
            break
    print (counter.__dict__)
    print (counter.keys)
    return np.array(z), y


###
x, y = parse_hexameter('data/metamorphoses/metamorphoses_hexameter.txt')
print (x, y)
