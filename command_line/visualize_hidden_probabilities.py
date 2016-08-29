from os import path
from os import listdir
from os.path import splitext
import numpy as np
from PIL import Image
import tensorflow as tf

__author__ = 'Annika Lindh'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('path', 'path_missing', 'Directory to where the debug data is located.')

if __name__ == '__main__':
    for dataFile in listdir(FLAGS.path):
        if dataFile.startswith('hidden_probabilities_') and dataFile.endswith('.npy'):
            data = np.load(path.join(FLAGS.path, dataFile))
            imgArr = .5 * (1 + np.tanh(data / 2.))
            imgArr = np.array(imgArr * 255, dtype=np.uint8)
            img = Image.frombuffer('L', [imgArr.shape[1], imgArr.shape[0]], imgArr.tostring(), 'raw', 'L', 0, 1)
            img.save(splitext(path.join(FLAGS.path, dataFile))[0] + ".png")
