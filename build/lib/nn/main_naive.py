import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.activation import sigmoid_double

def average_digit(data, digit):
  filtered_data = [x[0] for x in data if x[1][digit]]
  filtered_array = np.asarray(filtered_data)
  return np.average(filtered_array, axis=0)

def eval_single(x, W):
  return np.dot(np.transpose(W), x)

def predict(x, W, b, threshold):
  p = sigmoid_double(eval_single(x, W) + b)
  return True if p >= threshold else False

# def score((y_p, y_t), digit):
#   Y = np.argmax(y_t)
#   score = 0
#   if (y_p and Y == digit) or (not y_p and Y != digit):
#     score = 1
#   return score

# def evaluate(data, digit, threshold, W, b):
#   W = average_digit(data, digit)
#   preds = map(lambda x: predict(x, W, b, threshold), [d[0] for x in data])
#   scores = map(lambda p,t: score(p,t), zip(preds, [d[1] for x in data]))
#   return sum(scores) / len(scores)
