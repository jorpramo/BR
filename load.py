__author__ = 'jpradas'

import pymongo
import datetime
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def vocab_words(text):
    tokens=word_tokenize(text)
    vocab=['dados','jugador','fichas','prueba']
    spanish_stops = set(stopwords.words('spanish'))
    texto=[w.lower() for w in tokens if w in vocab and w not in spanish_stops]
    all_words=nltk.FreqDist(texto)
    return list(all_words)


spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

# Conexion con MongoDB
client = pymongo.MongoClient("localhost", 27017)
db = client.test
docs=db.DOCS

spanish_stops = set(stopwords.words('spanish'))

corpus_root = "C:/Users/jpradas/Documents/MASTER/TFM/code/data/"
newcorpus = PlaintextCorpusReader(corpus_root, '.*')
newcorpus.fileids()

for fileid in newcorpus.fileids():
    num_chars = len(newcorpus.raw(fileid))
    num_words = len(newcorpus.words(fileid))
    words = newcorpus.words(fileid)
    num_sents = len(newcorpus.sents(fileid))
    num_vocab = len(set(w.lower() for w in newcorpus.words(fileid)))
    # print(newcorpus.raw(fileid))
    bcf = BigramCollocationFinder.from_words(words)
    filter_stops = lambda w: len(w) < 3 or w in spanish_stops
    bcf.apply_word_filter(filter_stops)

    tag_bi=bcf.nbest(BigramAssocMeasures.likelihood_ratio, 5)

    post = {"nombre": fileid, "fecha": datetime.datetime.utcnow(),"tag_bi":tag_bi,"tag_vocab":vocab_words(newcorpus.raw(fileid)), "enc":0, "pos":0, "neg":0, "texto":newcorpus.raw(fileid), "tokenize":spanish_tokenizer.tokenize(newcorpus.raw(fileid)) }
    post_id = docs.insert_one(post).inserted_id
    print ( post_id )
