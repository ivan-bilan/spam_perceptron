# Code shared for perceptron training and testing.
import re

threshold = 0.6

def dot_product(features, weights):
  return sum(weights[f] for f in features)


def tokens(filename):
  """
  Read feature tokens
  Better results are achieved when splitting on
  whitespaces ("\s+") and not on non characters ("\W+")
  """
  with open(filename, 'r') as myfile:
    text = myfile.read().strip().lower()

  return re.split("\s+", text)


def prediction(features, weights):
  return dot_product(features, weights) > threshold


def filtered_tokens(filename, vocab):
  return [t for t in tokens(filename) if t in vocab]

