#!/usr/bin/env python
#
#                               kalavela.py
# 

def parse_kalevala(i_file):

    print ('TEXT FILE', i_file)
    START, END = 87, 23187
    w = []
    with open(i_file, 'r') as i_handle: 
        for i, line in enumerate(i_handle):
            if i >= START and i < END: 
                line = line.strip()
                if len(line) > 1:
                    w.append(line)
    return w


if __name__ == '__main__':

    STOPWORD = '"\:\;\“\”\.\,\(\)\!\?'
    IFILE   = 'data/kalevala/kalevala.txt'
    OFILE   = 'result/kalevala-tsne.png'
    MFILE   = 'model/kalevala-w2v.bin'

    from metamorphoses import bigram, make_w2v, scatter_plot
    """
    text = parse_kalevala(IFILE)
    spl_text = bigram(text, STOPWORD)

    model = make_w2v(spl_text)
    model.save(MFILE)
    del model
    """

    from gensim.models import word2vec
    model = word2vec.Word2Vec.load(MFILE)
    scatter_plot(model, OFILE)
