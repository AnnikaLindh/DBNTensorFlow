import tensorflow as tf
import os

import config

from yadlt.models.dbn_models.dbn import DeepBeliefNetwork
from yadlt.data_handlers.npy_data_handler import NpyDataHandler
from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_string('model_name', 'dbn', 'Name of the model.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', 1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'dbn/', 'Directory to store data relative to the algorithm.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the number of hidden units in the layers.')
flags.DEFINE_float('rbm_stddev', 0.0, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rates', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_sizes', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function: mean_squared or cross_entropy')
flags.DEFINE_string('regtype', 'none', 'Regularization type. none, l1 or l2')
flags.DEFINE_integer('l2reg', 5e-4, 'Regularization parameter when type is l2.')

# Data configuration
flags.DEFINE_integer('num_visible', 12500, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_string('visible_unit_type', 'gauss', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('hidden_unit_type', 'logistic', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('dataset_training', None, 'Npy file with the training data.')
flags.DEFINE_string('dataset_validation', None, 'Npy file with the validation data.')
flags.DEFINE_string('dataset_test', None, 'Npy file with the test data.')
flags.DEFINE_boolean('memory_mapping', True, 'default True, Whether to use memory mapping to avoid loading the full dataset into RAM.')
flags.DEFINE_boolean('load_batch', False, 'default False, Whether to load the current batch into RAM.')

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    # Conversion of RBM layers parameters from string to their specific type
    rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
    rbm_learning_rates = utilities.flag_to_list(FLAGS.rbm_learning_rates, 'float')
    rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
    rbm_batch_sizes = utilities.flag_to_list(FLAGS.rbm_batch_sizes, 'int')
    rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

    # Parameters validation
    assert len(rbm_layers) > 0

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    srbm = DeepBeliefNetwork(
        FLAGS.num_visible, rbm_layers, visible_unit_type=FLAGS.visible_unit_type,
        hidden_unit_type=FLAGS.hidden_unit_type, model_name=FLAGS.model_name,
        verbose=FLAGS.verbose, rbm_gibbs_k=rbm_gibbs_k, rbm_learning_rates=rbm_learning_rates,
        rbm_batch_sizes=rbm_batch_sizes, rbm_num_epochs=rbm_num_epochs, rbm_stddev=FLAGS.rbm_stddev,
        loss_func=FLAGS.loss_func, regtype=FLAGS.regtype, l2reg=FLAGS.l2reg,
        main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
        debug_hidden_units=100)

    # Prepare the data
    data_training = NpyDataHandler(FLAGS.dataset_training, FLAGS.memory_mapping, FLAGS.load_batch)
    data_validation = None if FLAGS.dataset_validation is None \
        else NpyDataHandler(FLAGS.dataset_validation, FLAGS.memory_mapping, FLAGS.load_batch)
    data_test = None if FLAGS.dataset_test is None \
        else NpyDataHandler(FLAGS.dataset_test, FLAGS.memory_mapping, FLAGS.load_batch)

    # Fit the model
    srbm.fit(data_training=data_training, data_validation=data_validation)

    # TODO Implement full reconstruction computation on test set
    #print('Test set accuracy: {}'.format(srbm.compute_accuracy(teX, teY)))
