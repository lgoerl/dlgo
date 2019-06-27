import gzip, os
import numpy as np
# import six.moves.cPickle as pickle

def encode_label(j):
  v = np.zeros(10)
  v[j] = 1.0
  return v

def shape_data(data):
  features = [np.reshape(x, (784,1)) for x in data[0]]
  labels = [encode_label(y) for y in data[1]]
  return list(zip(features, labels))

def load_data():
  path = os.path.join(os.path.expanduser('~'), 'Downloads/mnist.npz')
  data = np.load(path)
  x_train, y_train = data['x_train'], data['y_train']
  x_test, y_test = data['x_test'], data['y_test']
  data.close()
  return shape_data((x_train, y_train)), shape_data((x_test, y_test))
