Train a perception classifier to detect spam emails

Project structure:

 Training script: perceptron_training.py, 
 testing script:  perceptron_testing.py, 
 helper functions : perceptron_utils.py, 
 bash overhead for training with all parameters included: perceptron_training.sh

Input parameters:
-p : path to positive training data, 
-n : path to negative training data, 
-pp : path to positive test data, 
-nn : path to negative test data
-m : model path

Call example of the training script: perceptron_training.py -p=enron1/ham/ -n=enron1/spam/ -pp=enron2/ham/ -nn=enron2/spam/ -m=model1

Both scripts also display training and testing errors

Dependencies:
Python 2.7, 
nltk
 
 
Get the training an test sets from https://sites.google.com/site/rothbenj/teaching/repetitorium/perceptron.tgz?attredirects=0

