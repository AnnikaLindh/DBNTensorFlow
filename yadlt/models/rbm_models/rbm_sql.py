import os
from os.path import expanduser
import numpy as np
import tensorflow as tf
import mysql.connector.cursor as sql_cursor
from mysql.connector import Error as sql_error
import StringIO

from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


class RBMSQL:

    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    This class is based on the RBM class from blackecho's Deep-Learning-Tensorflow but modified to feed data from
    an SQL database for better memory handling capabilities.
    """

    def __init__(self, num_visible, num_hidden, visible_unit_type='bin', model_name='rbm_sql', verbose=1,
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10, num_epochs=10, stddev=0.1,
                 loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 main_dir='rbm_sql/', models_dir='models/', data_dir='data/', summary_dir='logs/'):

        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param visible_unit_type: type of the visible units, 'bin' (binary) or 'gauss' (gaussian)
        :param model_name: name of the model, used as filename. string, default 'dae'
        :param verbose: level of verbosity. optional, default 0
        :param gibbs_sampling_steps: optional, default 1
        :param learning_rate: Initial learning rate
        :param batch_size: Size of each mini-batch
        :param num_epochs: Number of epochs
        :param stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param loss_func: type of loss function
        :param regtype: regularization type
        :param l2reg: regularization parameter
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param models_dir: directory to store trained models
        :param data_dir: directory to store generated data
        :param summary_dir: directory to store tensorflow logs
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.model_name = model_name
        self.verbose = verbose

        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev

        self.loss_func = loss_func
        self.regtype = regtype
        self.l2reg = l2reg

        home = os.path.join(expanduser("~"), '.yadlt')
        self.main_dir = os.path.join(home, main_dir)
        self.models_dir = os.path.join(home, models_dir)
        self.data_dir = os.path.join(home, data_dir)
        self.tf_summary_dir = os.path.join(home, summary_dir)
        self.model_path = os.path.join(self.models_dir, self.model_name)

        print('Creating %s directory to save/restore models' % self.models_dir)
        utilities.create_dir(self.models_dir)
        print('Creating %s directory to save model generated data' % self.data_dir)
        utilities.create_dir(self.data_dir)
        print('Creating %s directory to save tensorboard logs' % self.tf_summary_dir)
        utilities.create_dir(self.tf_summary_dir)

        # tensorflow nodes
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_summary_writer_available = True
        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.input_data = None
        self.hrand = None
        self.vrand = None
        self.encode = None
        self.reconstruction = None
        self.cost = None
        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        # Retrieve run identifier
        run_id = 0
        for e in os.listdir(self.tf_summary_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r
        run_id += 1
        run_dir = os.path.join(self.tf_summary_dir, 'run' + str(run_id))
        print('Tensorboard logs dir for this run is %s' % run_dir)

        self.tf_summary_writer = tf.train.SummaryWriter(run_dir, self.tf_session.graph)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

    def fit(self, conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_training, stmt_validation=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_shuffle: Statement to generate random numbers used for batch assignment
        :param stmt_training: Prepared statement for selecting the data from the training set
        :param stmt_validation: Prepared statement for selecting the data from the validation set
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: tensorflow graph object
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model()
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_training, stmt_validation)
                self.tf_saver.save(self.tf_session, self.model_path)

    def _train_model(self, conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_training, stmt_validation=None):

        """ Train the model.
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_shuffle: Statement to generate random numbers used for batch assignment
        :param stmt_training: Prepared statement for selecting the data from the training set
        :param stmt_validation: Prepared statement for selecting the data from the validation set
        :return: self
        """

        for i in range(self.num_epochs):
            self._run_train_step(conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_training)

            if stmt_validation is not None:
                self._run_validation_error_and_summaries(i, conSelect, curSelect, stmt_validation)

    def _run_train_step(self, conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_fetch):

        """ Run a training step. A training step is made by randomly "shuffling" the training set by randomizing the
        values in the ORDER BY column, fetching the data in ordered batches and then run the variable update nodes for
        each batch.
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_shuffle: Prepared statement for generating random numbers for each row, used for batch assignment
        :param stmt_fetch: Prepared statement for retrieving 1 batch based on ORDER BY on the "shuffle" column
        :return: self
        """

        try:
            curInsert.execute(stmt_shuffle)
            conInsert.commit()
            curSelect.execute(stmt_fetch)

        except sql_error:
            print "Error: RBM failed to shuffle and prepare the data fetching."
            raise
        except:
            raise

        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        feed_dict = self._create_feed_dict(curSelect, self.batch_size)
        while feed_dict is not None:
            self.tf_session.run(updates, feed_dict=feed_dict['feed'])
            feed_dict = self._create_feed_dict(curSelect, self.batch_size)

        conSelect.commit()

    def _create_feed_dict(self, db_cursor, nRows):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
        :type db_cursor: sql_cursor.MySQLCursorPrepared
        :param db_cursor: Prepared MySQL cursor to the data table
        :param stmt_fetch: Prepared statement for selecting one batch of data
        :param offset: Offset to start the current data batch at
        :param nRows: Number of rows for the current data batch
        :return: a nested dictionary
            feed:(self.input_data: data, self.hrand: random_uniform, self.vrand: random_uniform)
            id_list: list of the ids corresponding to input_data in 'feed'
        """

        data = list()
        id_list = list()
        try:
            res = db_cursor.fetchmany(nRows)

            for currentRow in res:
                data.append( np.fromstring(currentRow[0], dtype=np.float32) )
                id_list.append(currentRow[1])

        except sql_error:
            print "Error: RBM failed to fetch data."
            raise
        except:
            raise
        else:
            if len(data) == 0:
                return None
            else:
                data = np.row_stack(data)
                return {
                    'feed': {
                        self.input_data: data,
                        self.hrand: np.random.rand(data.shape[0], self.num_hidden),
                        self.vrand: np.random.rand(data.shape[0], data.shape[1])
                    },
                    'id_list': id_list
                }

    def _run_validation_error_and_summaries(self, epoch, conSelect, curSelect, stmt_validation):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param curSelect: Prepared MySQL cursor for fetching data
        :param stmt_validation: Prepared statement for selecting the data from the validation set
        :return: self
        """
        try:
            curSelect.execute(stmt_validation)
        except sql_error:
            print "Error: RBM failed to prepare the validation  fetching."
            raise
        except:
            raise

        # TODO Loop and test all of the validation set?
        feed=self._create_feed_dict(curSelect, self.batch_size)['feed']
        try:
            result = self.tf_session.run([self.tf_merged_summaries, self.cost],
                                         feed_dict=feed)
            summary_str = result[0]
            err = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            err = self.tf_session.run(self.cost, feed_dict=feed)

        if self.verbose == 1:
            print("Reconstruction loss at step %s: %s" % (epoch, err))

        # TODO Run through the results properly
        curSelect.fetchall()
        conSelect.commit()

    def store_encodings(self, conSelect, conInsert, curSelect, curInsert, stmt_fetch, stmt_store, graph=None):

        """ Encode the data and store the encoded versions in the database
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_fetch: Prepared statement for selecting one batch of data
        :param stmt_store: Prepared statement for storing the encoded data
        :param graph: tf graph object, optional
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                try:
                    curSelect.execute(stmt_fetch)
                except sql_error:
                    print "Error: RBM failed to prepare the data fetching."
                    raise
                except:
                    raise

                feed_dict = self._create_feed_dict(curSelect, self.batch_size)
                while feed_dict is not None:
                    # Encode the current batch
                    encoded_data = self.tf_session.run(self.encode, feed_dict=feed_dict['feed'])
                    try:
                        # TODO Use separate connection for storing - make sure to set no locking
                        # Store the encoded version of each of the data samples in the current batch
                        for i in range(0, len(encoded_data)):
                            curInsert.execute( stmt_store, (StringIO.StringIO(encoded_data[i].tostring('C')),
                                                            feed_dict['id_list'][i],) )

                        conInsert.commit()
                    except sql_error:
                        print "Error: RBM failed to encode data for batch with IDs: ", feed_dict['id_list']
                        raise

                    feed_dict = self._create_feed_dict(curSelect, self.batch_size)

                conSelect.commit()

    def build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        :return: self
        """

        self._create_placeholders()
        self._create_variables()
        self.encode = self.sample_hidden_from_visible(self.input_data)[0]
        self.reconstruction = self.sample_visible_from_hidden(self.encode)

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.w_upd8 = self.W.assign_add(self.learning_rate * (positive - negative) / self.batch_size)
        self.bh_upd8 = self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hprobs0 - hprobs1, 0))
        self.bv_upd8 = self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - vprobs, 0))

        variables = [self.W, self.bh_, self.bv_]
        regterm = self.compute_regularization(variables)

        self._create_cost_function_node(vprobs, self.input_data, regterm=regterm)

    def compute_regularization(self, variables):
        """ Compute the regularization tensor.
        :param variables: list of model variables
        :return:
        """

        if self.regtype == 'none':
            return None

        else:
            regularizers = tf.constant(0.0)

            if self.regtype == 'l2':
                for v in variables:
                    regularizers = tf.add(regularizers, tf.nn.l2_loss(v))
            elif self.regtype == 'l1':
                for v in variables:
                    regularizers = tf.add(regularizers, tf.reduce_sum(tf.abs(v)))

            return tf.mul(self.l2reg, regularizers)

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, self.num_visible], name='x-input')
        self.hrand = tf.placeholder('float', [None, self.num_hidden], name='hrand')
        self.vrand = tf.placeholder('float', [None, self.num_visible], name='vrand')

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        self.W = tf.Variable(tf.truncated_normal(shape=[self.num_visible, self.num_hidden], stddev=0.1), name='weights')
        self.bh_ = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.constant(0.1, shape=[self.num_visible]), name='visible-bias')

    def _create_cost_function_node(self, data_reconstruction, data_actual, regterm=None):

        """ Create the cost function node.
        :param data_reconstruction: data as reconstructed by the model after encoding actual_data
        :param data_actual: the placeholder node for the actual activations of the visible units (pixel data)
        :param regterm: regularization term
        :return: self
        """

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                cost = - tf.reduce_mean(data_actual * tf.log(tf.clip_by_value(data_reconstruction, 1e-10, float('inf'))) +
                                        (1 - data_actual) * tf.log(tf.clip_by_value(1 - data_reconstruction, 1e-10, float('inf'))))

            elif self.loss_func == 'softmax_cross_entropy':
                softmax = tf.nn.softmax(data_reconstruction)
                cost = - tf.reduce_mean(data_actual * tf.log(softmax) + (1 - data_actual) * tf.log(1 - softmax))

            elif self.loss_func == 'mean_squared':
                cost = tf.sqrt(tf.reduce_mean(tf.square(data_actual - data_reconstruction)))

            else:
                cost = None

        if cost is not None:
            self.cost = (cost + regterm) if regterm is not None else cost
            _ = tf.scalar_summary(self.loss_func, self.cost)
        else:
            self.cost = None

    def gibbs_sampling_step(self, visible):

        """ Performs one step of gibbs sampling.
        :param visible: activations of the visible units
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """

        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.

        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """

        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
        hstates = utilities.sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units.
        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def load_model(self, num_visible, num_hidden, gibbs_sampling_steps, model_path):

        """ Load a trained model from disk. The shape of the model
        (num_visible, num_hidden) and the number of gibbs sampling steps
        must be known in order to restore the model.
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param gibbs_sampling_steps:
        :param model_path:
        :return: self
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.gibbs_sampling_steps = gibbs_sampling_steps

        self.build_model()

        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:
            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

    def get_model_parameters(self, graph=None, restore_previous_model=False):

        """ Return the model parameters in the form of numpy arrays.
        :param graph: tf graph object
        :return: model parameters
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                if restore_previous_model:
                    self.build_model()
                    self._initialize_tf_utilities_and_ops(restore_previous_model=True)
                self.tf_saver.restore(self.tf_session, self.model_path)

                return {
                    'W': self.W.eval(),
                    'bh_': self.bh_.eval(),
                    'bv_': self.bv_.eval()
                }
