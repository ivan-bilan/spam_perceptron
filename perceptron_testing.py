# -*- coding: utf-8 -*-

__date__ = "20.06.2016"

import os
import sys
import argparse
import json
import perceptron_utils

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--positive_dir', required = True)
  parser.add_argument('-n', '--negative_dir', required = True)
  parser.add_argument('-m', '--model', required = True)
  parser.add_argument('-s', '--filesuffix', default='.txt')
  opts = parser.parse_args()

  files_labels = \
    [(opts.positive_dir + "/" + f, 1) for f in os.listdir(opts.positive_dir) if f.endswith(opts.filesuffix)] + \
    [(opts.negative_dir + "/" + f, 0) for f in os.listdir(opts.negative_dir) if f.endswith(opts.filesuffix)]

  with open(opts.model, 'r') as modelfile:
    weights = json.load(modelfile)
  vocab = set(weights.keys())

  error_count = 0
  for input_file, desired_output in files_labels:
    features = set(perceptron_utils.filtered_tokens(input_file, vocab))
    result = perceptron_utils.prediction(features, weights)
    if desired_output != result:
      error_count += 1
  print "test errors: ", error_count
  print "number of instances: ", len(files_labels)
  
if __name__ == "__main__":
   main(sys.argv[1:])

