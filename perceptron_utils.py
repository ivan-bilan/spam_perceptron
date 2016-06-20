# -*- coding: utf-8 -*-

__date__ = "20.06.2016"

# Code shared for perceptron training and testing.
import re
import codecs
import nltk
from itertools import chain
from nltk import word_tokenize
from nltk.util import ngrams

threshold = 0.6

def dot_product(features, weights):
  return sum(weights[f] for f in features)


def tokens(filename):
  """
  Read feature tokens
  """
  with codecs.open(filename, 'rb', encoding="windows-1251") as myfile:
    text = myfile.read().strip().lower()

  token = nltk.word_tokenize(unicode(text))
  unigrams = ngrams(token, 1)
  trigrams = ngrams(token, 3)

  # unigrams = re.split(r"\s+", text)

  return chain(unigrams, trigrams)


def prediction(features, weights):
  return dot_product(features, weights) > threshold


def filtered_tokens(filename, vocab):
  return [t for t in tokens(filename) if t in vocab]

