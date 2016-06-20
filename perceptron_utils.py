# -*- coding: utf-8 -*-

__date__ = "20.06.2016"

# Code shared for perceptron training and testing.
import re
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

threshold = 0.6

def dot_product(features, weights):
  return sum(weights[f] for f in features)


def tokens(filename):
  """
  Read feature tokens
  Better results are achieved when splitting on
  whitespaces ("\s+") and not on none characters ("\W+")
  """
  with open(filename, 'r') as myfile:
    text = myfile.read().strip().lower()

  # token = nltk.word_tokenize(unicode(text))
  # bigrams = ngrams(token,2)

  unigrams = re.split(r"\s+", text)

  # bigrams = re.split(r"\b\w+\s\w+", text)

  return unigrams


def prediction(features, weights):
  return dot_product(features, weights) > threshold


def filtered_tokens(filename, vocab):
  return [t for t in tokens(filename) if t in vocab]

