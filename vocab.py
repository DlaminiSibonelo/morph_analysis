# @Author: Sibonelo Dlamini

import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.corpus import brown


def load_corpus():
    corpus_root = r"c:/corpus"
    corpus = PlaintextCorpusReader(corpus_root, '.*')
    print('Corpus loaded...')

    return corpus

def to_lowercase(corpus):
    corpus_sents = corpus.sents()

    new_sents = []
    for sent in corpus_sents:
        words = [w.lower() for w in sent]
        new_sents.append(words)

    return new_sents

def remove_stop_words(corpus_words):
    stop_words = ['.', ',', '"', '-', '?', '\'', '“', '."', ':', '!', '(', ')']
    words = [word.lower() for word in corpus_words if word.isalnum()]
    return words

def is_num(string):
    try:
        float(string)
        return True
    except ValueError as e:
        return False

def get_vocab(corpus, limit=False):
    corpus_words = corpus.words()

    if limit:
        corpus_words = corpus_words[:1000000]

    size_all = len(corpus_words)
    print('Number of all tokens: {:,}\n'.format(size_all))

    corpus_words_alnum = remove_stop_words(corpus_words) 
    print('Number of alphanumeric tokens: {:,}\n'.format(len(corpus_words_alnum)))

    words = [word for word in corpus_words_alnum if not (word.isalnum() or word.is_num())]

    vocab = sorted(set(words))

    print('Size of vocabulary: {}'.format(len(vocab)))
    for word in vocab:
        print(word)

    return set(corpus_words_alnum)

corpus = load_corpus()
get_vocab(corpus, False)
