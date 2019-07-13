import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks import small
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
# from kers.optimizers import Adagrad, Adadelta, SGD

go_board_rows, go_board_cols = 19,19
num_classes = go_board_cols*go_board_rows

NUM_GAMES = 1000
EPOCHS = 5
BATCH_SIZE = 128
model = Sequential()
optimizer = Adadelta()
encoder_name = 'sevenplane'
# SGD(lr=0.1, momentum=0.9, decay=0.01)
# Adagrad()


def load_generators(num_games, encoder_name):
    processor = GoDataProcessor(encoder=encoder_name)
    generator = processor.load_go_data('train', NUM_GAMES, use_generator=True)
    test_generator = processor.load_go_data('test', NUM_GAMES, use_generator=True)
    return generator, test_generator

def compile_model(encoder, optimizer, network):
    # encoder = get_encoder_by_name(encoder_name, go_board_cols)
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = network.layers(input_shape)

    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def fit(model, train, test):
    model.fit_generator(
        generator=train.generate(BATCH_SIZE, num_classes),
        epochs=EPOCHS,
        steps_per_epoch=train.get_num_samples()/BATCH_SIZE,
        validation_data=test.generate(BATCH_SIZE, num_classes),
        validation_steps=test.get_num_samples()/BATCH_SIZE,
        callbacks=[
            ModelCheckpoint('./dlgo/checkpoints/small_model_epoch{epoch}.hdf5')
        ]
    )

    model.evaluate_generator(
        generator=test.generate(BATCH_SIZE, num_classes),
        steps=test.get_num_samples()/BATCH_SIZE
    )

if __name__ == '__main__':
    encoder = get_encoder_by_name(encoder_name, go_board_cols)
    train_generator, test_generator = load_generators(num_games=NUM_GAMES, encoder_name='sevenplane')
    model = compile_model(encoder=encoder, optimizer=optimizer, network=small)
    fit(model, train_generator, test_generator)
