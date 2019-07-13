import h5py, os
from glob import glob
from keras.optimizers import Adadelta

from dlgo.agent.dlagent import DeepLearningAgent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks import small
from dlgo.nn.main_network import compile_model

def get_most_recent_checkpoint(prefix='./dlgo/checkpoints'):
    pattern = f'{prefix}/*h5'
    checkpoints = glob(pattern)
    return max(checkpoints, key=os.path.getctime)

if __name__ == '__main__':
    optimizer = Adadelta()
    encoder = get_encoder_by_name('sevenplane', 19)
    model = compile_model(encoder, optimizer, small)
    model.load_weights(get_most_recent_checkpoint())
    bot = DeepLearningAgent(model, encoder)
    serialized_model = h5py.File('./dlgo/agent/latest.hdf5', 'w')
    bot.serialize(serialized_model)