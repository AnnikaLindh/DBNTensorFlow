import numpy as np
import tensorflow as tf
import os
import mysql.connector as sql
from mysql.connector import errorcode as sql_err

import config

from yadlt.models.rbm_models import unsupervised_dbn
from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_string('model_name', 'puredbn', 'Name of the model.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', 1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'puredbn/', 'Directory to store data relative to the algorithm.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the number of hidden units in the layers.')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rates', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_sizes', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function: mean_squared or cross_entropy')
flags.DEFINE_string('regtype', 'none', 'Regularization type. none, l1 or l2')
flags.DEFINE_integer('l2reg', 5e-4, 'Regularization parameter when type is l2.')

# Data configuration
flags.DEFINE_integer('num_visible', 12500, 'Number of visible units must equal number of features in the data.')
flags.DEFINE_string('visible_unit_type', 'gauss', 'Visible unit type. gauss or bin')
flags.DEFINE_string('db_host', '', 'Database hostname/IP.')
flags.DEFINE_integer('db_port', '', 'Database port.')
flags.DEFINE_string('db_user', '', 'Database username. The user needs permissions SELECT, UPDATE on the data table.')
flags.DEFINE_string('db_pass', '', 'Password for the database user.')
flags.DEFINE_string('db_name', '', 'Name of the database.')
flags.DEFINE_string('db_table_training', '', 'Table containing the training set.')
flags.DEFINE_string('db_table_validation', None, 'Table containing the validation set.')
flags.DEFINE_string('db_col_data', '', 'Column containing the input data.')
flags.DEFINE_string('db_col_encoded', '', 'Column to hold the encoded data.')
flags.DEFINE_string('db_col_id', '', 'Unique ID-column.')
flags.DEFINE_string('db_col_shuffle', '', 'Column to hold randomly generated values for batch assignment.')

# Database queries
stmt_base_init_encoded_column = 'UPDATE %s SET %s = %s'
stmt_seed_rand = 'SELECT RAND(?)'
stmt_base_shuffle = 'UPDATE %s SET %s = RAND()'
stmt_base_fetch = 'SELECT %s, %s FROM %s ORDER BY %s'
stmt_base_store = 'UPDATE %s SET %s = ? WHERE %s = ?'

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

    # TODO Make sure I call the PureDBN with a proper rbm_stddev
    srbm = unsupervised_dbn.UnsupervisedDeepBeliefNetwork(
        FLAGS.num_visible, rbm_layers, visible_unit_type=FLAGS.visible_unit_type, model_name=FLAGS.model_name,
        verbose=FLAGS.verbose, rbm_gibbs_k=rbm_gibbs_k, rbm_learning_rates=rbm_learning_rates,
        rbm_batch_sizes=rbm_batch_sizes, rbm_num_epochs=rbm_num_epochs, rbm_stddev=FLAGS.rbm_stddev,
        loss_func=FLAGS.loss_func, regtype=FLAGS.regtype, l2reg=FLAGS.l2reg,
        main_dir=FLAGS.main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir)

    # Create the database query statements
    stmt_training_init_encoded_column = stmt_base_init_encoded_column % (FLAGS.db_table_training,
                                                                         FLAGS.db_col_encoded, FLAGS.db_col_data,)
    stmt_training_shuffle = stmt_base_shuffle % (FLAGS.db_table_training, FLAGS.db_col_shuffle,)
    stmt_training_fetch_first = stmt_base_fetch % (FLAGS.db_col_data, FLAGS.db_col_id, FLAGS.db_table_training,
                                             FLAGS.db_col_shuffle,)
    stmt_training_fetch = stmt_base_fetch % (FLAGS.db_col_encoded, FLAGS.db_col_id, FLAGS.db_table_training,
                                             FLAGS.db_col_shuffle,)
    stmt_training_store = stmt_base_store % (FLAGS.db_table_training, FLAGS.db_col_encoded, FLAGS.db_col_id,)

    if FLAGS.db_table_validation is not None:
        stmt_validation_init_encoded_column = stmt_base_init_encoded_column % (FLAGS.db_table_validation,
                                                                               FLAGS.db_col_encoded, FLAGS.db_col_data,)
        stmt_validation_fetch_first = stmt_base_fetch % (FLAGS.db_col_data, FLAGS.db_col_id, FLAGS.db_table_validation,
                                                   FLAGS.db_col_shuffle,)
        stmt_validation_fetch = stmt_base_fetch % (FLAGS.db_col_encoded, FLAGS.db_col_id, FLAGS.db_table_validation,
                                                   FLAGS.db_col_shuffle,)
        stmt_validation_store = stmt_base_store % (FLAGS.db_table_validation, FLAGS.db_col_encoded, FLAGS.db_col_id,)
    else:
        stmt_validation_init_encoded_column = stmt_validation_fetch_first = stmt_validation_fetch = stmt_validation_store = None

    # Setup the database connection
    try:
        conSelect = sql.connect(user=FLAGS.db_user, password=FLAGS.db_pass, host=FLAGS.db_host, database=FLAGS.db_name)
        conInsert = sql.connect(user=FLAGS.db_user, password=FLAGS.db_pass, host=FLAGS.db_host, database=FLAGS.db_name)
        curSelect = conSelect.cursor(prepared=True)
        curInsert = conInsert.cursor(prepared=True)
        curSelect.execute(stmt_seed_rand, (FLAGS.seed,))
        curSelect.fetchall() # Clear the cursor before next call
        conSelect.commit()
    except sql.Error as err:
        if err.errno == sql_err.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == sql_err.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        raise
    except:
        raise

    # Fit the model

    srbm.fit(conSelect=conSelect, conInsert=conInsert, curSelect=curSelect, curInsert=curInsert, stmt_shuffle=stmt_training_shuffle,
             stmt_fetch_train_first=stmt_training_fetch_first, stmt_fetch_train=stmt_training_fetch, stmt_store_train=stmt_training_store,
             stmt_fetch_validation_first=stmt_validation_fetch_first, stmt_fetch_validation=stmt_validation_fetch, stmt_store_validation=stmt_validation_store)

    # Clean up DB connections
    curInsert.close()
    curSelect.close()
    conInsert.close()
    conSelect.close()

    # TODO Implement full reconstruction computation on test set
    #print('Test set accuracy: {}'.format(srbm.compute_accuracy(teX, teY)))
