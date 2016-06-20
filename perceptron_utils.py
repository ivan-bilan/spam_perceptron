# Code shared for perceptron training and testing.
import re

threshold = 0.5

def dot_product(features, weights):
  return sum(weights[f] for f in features)

def tokens(filename):
  with open(filename, 'r') as myfile:
    text=myfile.read().strip().lower()
  return re.split("\W+", text)

def prediction(features, weights):
  return dot_product(features, weights) > threshold

def filtered_tokens(filename, vocab):
  return [t for t in tokens(filename) if t in vocab]

