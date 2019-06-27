import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

np.random.seed(123)

# data loading and preprocessing
x = np.load('../generated_games/features-40k.npy')
y = np.load('../generated_games/labels-40k.npy')
samples = x.shape[0]
size = 9*9
input_shape = (size, size)

x = x.reshape(samples, size, size, 1)
# y = y.reshape(samples, size, size, 1)

train_samples = int(0.9 * samples)
x_train, x_test = x[:train_samples], x[train_samples:]
y_train, y_test = y[:train_samples], y[train_samples:]

model = Sequential()

model.add(Conv2D(
  filters=48,
  kernel_size=(3,3),
  activation='relu',
  padding='same',
  input_shape=input_shape,
))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(48, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(size*size, activation='softmax'))
model.summary()


model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy'],
)

model.fit(
  x_train, y_train,
  batch_size=64,
  epochs=100,
  verbose=1,
  validation_data=(x_test, y_test)
)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
