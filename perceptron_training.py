# -*- coding: utf-8 -*-

__date__ = "20.06.2016"

import os
import random
import sys
import argparse
import json
import perceptron_utils
from collections import Counter


learning_rate = 0.01
vocab_size = 20000
training_iterations = 50
random.seed(0)


def vocabulary(filelist, vocabsize):
  token_counter = Counter()
  for filename in filelist:
    for token in perceptron_utils.tokens(filename):
      token_counter[token] += 1
  return set([tc[0] for tc in token_counter.most_common(vocabsize)])

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-pp', '--positive_dir_test', required = True)
  parser.add_argument('-nn', '--negative_dir_test', required = True)
  parser.add_argument('-p', '--positive_dir', required = True)
  parser.add_argument('-n', '--negative_dir', required = True)
  parser.add_argument('-m', '--model', required = True)
  parser.add_argument('-s', '--filesuffix', default='.txt')

  opts = parser.parse_args()

  # read training data
  files_labels = \
    [(opts.positive_dir + "/" + f, 1) for f in os.listdir(opts.positive_dir) if f.endswith(opts.filesuffix)] + \
    [(opts.negative_dir + "/" + f, 0) for f in os.listdir(opts.negative_dir) if f.endswith(opts.filesuffix)]
  random.shuffle(files_labels)
  filelist = [fl[0] for fl in files_labels]

  # read test data
  files_labels_test = \
    [(opts.positive_dir_test + "/" + f, 1) for f in os.listdir(opts.positive_dir_test) if f.endswith(opts.filesuffix)] + \
    [(opts.negative_dir_test + "/" + f, 0) for f in os.listdir(opts.negative_dir_test) if f.endswith(opts.filesuffix)]

  vocab = vocabulary(filelist, vocab_size)
  weights = {}
  for token in vocab:
    weights[token] = 0.0

  for i in xrange(training_iterations):
    error_count = 0
    for input_file, desired_output in files_labels:
      features = set(perceptron_utils.filtered_tokens(input_file, vocab))
      result = perceptron_utils.prediction(features, weights)
      error = desired_output - result
      if error != 0:
        error_count += 1
        for f in features:
          weights[f] += learning_rate * error
    print('-' * 20)
    print "iteration: ", i + 1
    print "train errors: ", error_count
    error_count = 0
    for input_file, desired_output in files_labels_test:
      features = set(perceptron_utils.filtered_tokens(input_file, vocab))
      result = perceptron_utils.prediction(features, weights)
      if desired_output != result:
        error_count += 1
    print "test errors: ", error_count

  print "number of train instances: ", len(files_labels)
  print "number of test instances: ", len(files_labels_test)

  with open(opts.model, 'w') as modelfile:
    json.dump(weights, modelfile)


if __name__ == "__main__":
   main(sys.argv[1:])

