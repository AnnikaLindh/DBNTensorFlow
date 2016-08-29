import os
import tensorflow as tf
from yadlt.models.rbm_models import unsupervised_dbn
import config
from yadlt.utils import utilities

__author__ = 'Annika Lindh'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'puredbn', 'Name of the model.')
flags.DEFINE_string('main_dir', 'puredbn/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('num_visible', 12500, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_string('visible_unit_type', 'gauss', 'Visible unit type. gauss or bin')
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the number of hidden units in the layers.')

if __name__ == '__main__':
    rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    srbm = unsupervised_dbn.UnsupervisedDeepBeliefNetwork(
        FLAGS.num_visible, rbm_layers, visible_unit_type=FLAGS.visible_unit_type, model_name=FLAGS.model_name,
        main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)

    srbm.export_filters([50,50], 5, 4)
