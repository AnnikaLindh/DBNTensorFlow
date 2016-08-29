import os
from os.path import basename, splitext
import tensorflow as tf
from yadlt.models.dbn_models.dbn import DeepBeliefNetwork
from yadlt.models.dbn_models.dbn_finetuned import DBNFinetuned
from yadlt.data_handlers.npy_data_handler import NpyDataHandler
import config
from yadlt.utils import utilities

__author__ = 'Annika Lindh'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'name_missing', 'Name of the model.')
flags.DEFINE_string('main_dir', 'dir_missing/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('num_visible', 5120, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_string('visible_unit_type', 'logistic', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('hidden_unit_type', 'bin', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('rbm_layers', '0,', 'Comma-separated values for the number of hidden units in the layers.')
flags.DEFINE_string('datafiles', 'filename_missing,', 'Comma-separated names of the input files.')
flags.DEFINE_string('rbm_batch_sizes', '10,', 'Size of each mini-batch.')
flags.DEFINE_boolean('memory_mapping', True, 'default True, Whether to use memory mapping to avoid loading the full dataset into RAM.')
flags.DEFINE_boolean('load_batch', False, 'default False, Whether to load the current batch into RAM.')
flags.DEFINE_boolean('finetuned', False, 'default False, Whether to load a finetuned model.')

if __name__ == '__main__':
    rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
    datafiles = utilities.flag_to_list(FLAGS.datafiles, 'str')

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    if FLAGS.finetuned:
        rbm_layers = [FLAGS.num_visible] + rbm_layers
        model = DBNFinetuned(
            layers=rbm_layers, visible_unit_type=FLAGS.visible_unit_type, hidden_unit_type=FLAGS.hidden_unit_type,
            model_name=FLAGS.model_name,
            main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)
    else:
        model = DeepBeliefNetwork(
            FLAGS.num_visible, rbm_layers, visible_unit_type=FLAGS.visible_unit_type,
            hidden_unit_type=FLAGS.hidden_unit_type, model_name=FLAGS.model_name,
            main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)

    model.load_model()

    for dataFile in datafiles:
        data = NpyDataHandler(dataFile, FLAGS.memory_mapping, FLAGS.load_batch)
        model.store_encodings(data, splitext(basename(dataFile))[0])
