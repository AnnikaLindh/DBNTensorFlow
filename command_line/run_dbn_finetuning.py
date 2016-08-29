import os
import tensorflow as tf
from yadlt.models.dbn_models.dbn import DeepBeliefNetwork
from yadlt.models.dbn_models.dbn_finetuned import DBNFinetuned
from yadlt.data_handlers.npy_data_handler import NpyDataHandler
import config
from yadlt.utils import utilities

__author__ = 'Annika Lindh'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('stored_model_name', 'dbn', 'Name of the model.')
flags.DEFINE_string('finetuning_model_name', 'dbn_finetuned', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('num_visible', 2000, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_string('visible_unit_type', 'logistic', 'Visible unit type. gauss, bin, logistic or relu')
flags.DEFINE_string('hidden_unit_type', 'bin', 'Hidden unit type. gauss, bin, logistic or relu')
flags.DEFINE_float('stddev', 0.0, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_float('learning_rate', '0.01', 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', '30', 'Number of epochs.')
flags.DEFINE_integer('batch_size', '10', 'Size of each mini-batch.')
flags.DEFINE_integer('gibbs_sampling_steps', '1', 'Gibbs sampling steps.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function: mean_squared or cross_entropy')
flags.DEFINE_string('regtype', 'none', 'Regularization type. none, l1 or l2')
flags.DEFINE_integer('l2reg', 5e-4, 'Regularization parameter when type is l2.')
flags.DEFINE_string('rbm_layers', '100,', 'Comma-separated values for the number of hidden units in the layers.')
flags.DEFINE_boolean('restore_previous_model', True, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_string('dataset_training', None, 'Npy file with the training data.')
flags.DEFINE_string('dataset_validation', None, 'Npy file with the validation data.')
flags.DEFINE_string('dataset_test', None, 'Npy file with the test data.')
flags.DEFINE_boolean('memory_mapping', True, 'default True, Whether to use memory mapping to avoid loading the full dataset into RAM.')
flags.DEFINE_boolean('load_batch', False, 'default False, Whether to load the current batch into RAM.')

if __name__ == '__main__':
    rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')

    stored_models_dir = os.path.join(config.models_dir, FLAGS.stored_model_name)
    stored_data_dir = os.path.join(config.data_dir, FLAGS.stored_model_name)
    stored_summary_dir = os.path.join(config.summary_dir, FLAGS.stored_model_name)

    finetuned_models_dir = os.path.join(config.models_dir, FLAGS.finetuning_model_name)
    finetuned_data_dir = os.path.join(config.data_dir, FLAGS.finetuning_model_name)
    finetuned_summary_dir = os.path.join(config.summary_dir, FLAGS.finetuning_model_name)


    srbm = DeepBeliefNetwork(
        FLAGS.num_visible, rbm_layers, model_name=FLAGS.stored_model_name,
        visible_unit_type=FLAGS.visible_unit_type, hidden_unit_type=FLAGS.hidden_unit_type,
        main_dir=FLAGS.stored_model_name, models_dir=stored_models_dir, data_dir=stored_data_dir, summary_dir=stored_summary_dir)

    dbn_finetuned = srbm.get_finetuning_DBN(
        model_name=FLAGS.finetuning_model_name, main_dir=FLAGS.finetuning_model_name,
        gibbs_sampling_steps=FLAGS.gibbs_sampling_steps, learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs, stddev=FLAGS.stddev, loss_func=FLAGS.loss_func, regtype=FLAGS.regtype,
        l2reg=FLAGS.l2reg, models_dir=finetuned_models_dir,data_dir=stored_data_dir,summary_dir=stored_summary_dir,
        restore_previous_model=FLAGS.restore_previous_model)

    # Prepare the data
    data_training = NpyDataHandler(FLAGS.dataset_training, FLAGS.memory_mapping, FLAGS.load_batch)
    data_validation = None if FLAGS.dataset_validation is None \
        else NpyDataHandler(FLAGS.dataset_validation, FLAGS.memory_mapping, FLAGS.load_batch)
    data_test = None if FLAGS.dataset_test is None \
        else NpyDataHandler(FLAGS.dataset_test, FLAGS.memory_mapping, FLAGS.load_batch)

    # Fit the model
    dbn_finetuned.fit(data_training=data_training, data_validation=data_validation)
