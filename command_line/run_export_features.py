import os
import tensorflow as tf
from yadlt.models.dbn_models.dbn import DeepBeliefNetwork
from yadlt.models.dbn_models.dbn_finetuned import DBNFinetuned
import config
from yadlt.utils import utilities

__author__ = 'Annika Lindh'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'dbn', 'Name of the model.')
flags.DEFINE_string('main_dir', 'dbn/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('num_visible', 2000, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_integer('num_img_filters', 5, 'Number of filters.')
flags.DEFINE_integer('img_side', 20, 'Side length of each image.')
flags.DEFINE_integer('padding', 4, 'Padding around each generated feature.')
flags.DEFINE_integer('features_per_row', 10, 'How many features (consisting of num_img_filters each) per row.')
flags.DEFINE_string('visible_unit_type', 'logistic', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('hidden_unit_type', 'bin', 'Hidden unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('rbm_layers', '100,', 'Comma-separated values for the number of hidden units in the layers.')
flags.DEFINE_boolean('finetuned', False, 'default False, Whether to load a finetuned model.')

if __name__ == '__main__':
    rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    if FLAGS.finetuned:
        rbm_layers = [FLAGS.num_visible] + rbm_layers
        model = DBNFinetuned(
            layers=rbm_layers, visible_unit_type=FLAGS.visible_unit_type, hidden_unit_type=FLAGS.hidden_unit_type,
            model_name=FLAGS.model_name,
            main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)

        model.load_model()
    else:
        model = DeepBeliefNetwork(
            FLAGS.num_visible, rbm_layers, model_name=FLAGS.model_name,
            visible_unit_type=FLAGS.visible_unit_type, hidden_unit_type=FLAGS.hidden_unit_type,
            main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)

    model.export_features([FLAGS.img_side, FLAGS.img_side], FLAGS.num_img_filters, FLAGS.features_per_row, FLAGS.padding)
