import numpy as np

def sigmoid_double(x):
  return 1/(1 + np.exp(-x))

def sigmoid(v):
  return np.vectorize(sigmoid_double)(v)

def sigmoid_prime_double(x):
  return sigmoid_double(x) * (1 - sigmoid_double(x))

def sigmoid_prime(v):
  return np.vectorize(sigmoid_prime_double)(v)
