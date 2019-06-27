import numpy as np

def encode_label(j):
  v = np.zeros(10)
  v[j] = 1.0
  return v

def shape_data(data):
  # data is otf an iterable collection of (img_pixels, label)
  return [(
    np.array(d[0]).reshape(784,1),
    encode_label(d[1]),
  ) for d in data]

def load_data():
  with gzip.open('mnist.pkl.gz', 'rb') as f:
    train, validation, test = pickle.load(f)
  return shape_data(train), shape_data(test)
