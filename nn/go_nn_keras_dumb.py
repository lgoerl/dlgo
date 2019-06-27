import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(123)

# data loading and preprocessing
x = np.load('../generated_games/features-40k.npy')
y = np.load('../generated_games/labels-40k.npy')
samples = x.shape[0]
board_size = 9*9

x = x.reshape(samples, board_size)
y = y.reshape(samples, board_size)

train_samples = int(0.9 * samples)
x_train, x_test = x[:train_samples], x[train_samples:]
y_train, y_test = y[:train_samples], y[train_samples:]

# model definition
model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(board_size,)))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(board_size, activation='sigmoid'))
model.summary()

# model compiling
model.compile(
  loss='mean_squared_error',
  optimizer='sgd',
  metrics=['accuracy'],
)

model.fit(
  x_train, y_train,
  batch_size=64,
  epochs=15,
  verbose=1,
  validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
