import os
from os.path import expanduser
import tensorflow as tf
import numpy as np

from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


class RBM:

    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    This class is based on the RBM class from blackecho's Deep-Learning-Tensorflow.
    """

    def __init__(self, num_visible, num_hidden, visible_unit_type='logistic', hidden_unit_type='bin',
                 model_name='rbm', verbose=1,
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10, num_epochs=10, stddev=0.01,
                 loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 main_dir='rbm/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 encodings_dir='encodings/', debug_dir='debug/', debug_hidden_units=0):

        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param visible_unit_type: type of the visible units, logistic, bin, gauss or relu
        :param hidden_unit_type: type of the hidden units, logistic, bin, gauss or relu
        :param model_name: name of the model, used as filename. string, default 'rbm'
        :param verbose: level of verbosity. optional, default 1
        :param gibbs_sampling_steps: optional, default 1
        :param learning_rate: Initial learning rate
        :param batch_size: Number of samples in one mini-batch
        :param num_epochs: Number of epochs
        :param stddev: optional, default 0.01. Ignored if visible_unit_type is not 'gauss'
        :param loss_func: type of loss function
        :param regtype: regularization type
        :param l2reg: regularization parameter
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param models_dir: directory to store trained models
        :param data_dir: directory to store generated data
        :param summary_dir: directory to store tensorflow logs
        :param encodings_dir: directory to store final encoded versions of the data
        :param debug_dir: directory to store debug data
        :param debug_hidden_units: number of neurons for which to visualise hidden probabilities
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.hidden_unit_type = hidden_unit_type
        model_name = model_name
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
        self.model_path = os.path.join(self.models_dir, model_name)
        self.encodings_path = os.path.join(self.models_dir, encodings_dir)
        self.debug_path = os.path.join(self.models_dir, debug_dir)
        self.debug_hidden_units = min(debug_hidden_units, num_hidden)

        utilities.create_dir(self.models_dir)
        utilities.create_dir(self.data_dir)
        utilities.create_dir(self.encodings_path)
        utilities.create_dir(self.tf_summary_dir)
        utilities.create_dir(self.debug_path)

        # tensorflow nodes
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer_train = None
        self.tf_summary_writer_validation = None
        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.input_data = None
        self.encode = None
        self.reconstruction = None
        self.cost_cross_entropy = None
        self.cost_mean_squared = None
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
        if self.verbose >= 1:
            print('Tensorboard logs dir for this run is %s' % run_dir)

        self.tf_summary_writer_train = tf.train.SummaryWriter(run_dir + '/train', self.tf_session.graph)
        self.tf_summary_writer_validation = tf.train.SummaryWriter(run_dir + '/validation', self.tf_session.graph)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

    def fit(self, data_training, data_validation=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param data_training: Object that provides mini-batches for feeding and a function to save the encoded version
        :param data_validation: Object that provides mini-batches for feeding and a function to save the encoded version
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
                self._train_model(data_training, data_validation)
                self.tf_saver.save(self.tf_session, self.model_path)

    def _train_model(self, data_training, data_validation=None):

        """ Train the model.
        :param data_training: Object that provides mini-batches for feeding and a function to save the encoded version
        :param data_validation: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

        if self.debug_hidden_units > 0:
            if self.verbose >= 2:
                print "Storing initial hidden probabilities."
            self._store_hidden_probabilities(0, data_training, self.debug_hidden_units)

        if self.verbose >= 2:
            print "Running initial error calculations."

        self._output_costs(0, data_training, 'training')
        if data_validation is not None:
            self._output_costs(0, data_validation, 'validation')

        for i in range(1, self.num_epochs+1):
            if self.verbose >= 2:
                print "Starting epoch."
            data_training.startEpoch()
            if self.verbose >= 2:
                print "Running training step."
            stopEarly = self._run_train_step(data_training)

            if self.debug_hidden_units > 0:
                self._store_hidden_probabilities(i, data_training, self.debug_hidden_units)

            if i % 10 == 0:
                if self.verbose >= 2:
                    print "Running error calculations."

                self._output_costs(i, data_training, 'training')
                if data_validation is not None:
                    self._output_costs(i, data_validation, 'validation')

            if stopEarly:
                break

    def _run_train_step(self, data):

        """ Run a training step. A training step is made by randomly shuffling the dataset, fetching the data
        one mini-batch at a time, and run the variable update nodes for each batch.
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        feed_dict = self._create_feed_dict(data, self.batch_size)
        while feed_dict is not None:
            updated_values = self.tf_session.run(updates, feed_dict=feed_dict)

            # Check for NaN values due to exploding weights (if this happens, try lowering the learning rate)
            hasNan = False
            for iValue in range(0, len(updated_values)):
                if np.isfinite(updated_values[iValue]).all() == False:
                    hasNan = True
                    print "WARNING: %s have NaN values." % ("Weights" if iValue == 0 else
                                                            "Hidden biases" if iValue == 1 else
                                                            "Visible biases")
            if hasNan:
                feed_dict = None
                return True
            else:
                feed_dict = self._create_feed_dict(data, self.batch_size)

        return False

    def _create_feed_dict(self, datahandler, batchSize):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
        :param datahandler: A datahandler object that can provide mini-batches of data samples.
        :param batchSize: Number of training instances for this feed batch
        :return: a dictionary: {
                                self.input_data: data,
                                }
        """

        data = datahandler.nextBatch(batchSize)
        if data is None:
            return None
        else:
            return {
                self.input_data: data
            }

    def _run_error_and_summaries(self, epoch, data, writer):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

        if self.loss_func == None:
            return

        cost_names = list()
        cost_updates = list()
        if self.loss_func == 'cross_entropy' or self.loss_func == 'all':
            cost_updates.append(self.cost_cross_entropy)
            cost_names.append('cross_entropy')
        if self.loss_func == 'mean_squared' or self.loss_func == 'all':
            cost_updates.append(self.cost_mean_squared)
            cost_names.append('mean_squared')

        numBatches = 0
        totalBatchCosts = list()
        for _ in cost_updates:
            totalBatchCosts.append(0.0)

        data.startEpoch(shuffle=False)
        feed = self._create_feed_dict(data, self.batch_size)
        while feed is not None:
            current_costs = self.tf_session.run(cost_updates, feed_dict=feed)
            for i in range(0, len(totalBatchCosts)):
                totalBatchCosts[i] += current_costs[i]

            numBatches += 1
            feed = self._create_feed_dict(data, self.batch_size)

        for i in range(0, len(totalBatchCosts)):
            avgCost = totalBatchCosts[i] / numBatches
            if self.verbose >= 1:
                print "\t" + cost_names[i] + ': ' + str(avgCost)
            try:
                summary_str = tf.Summary(value=[tf.Summary.Value(tag=cost_names[i], simple_value=avgCost)])
                writer.add_summary(summary_str, epoch)
            except tf.errors.InvalidArgumentError:
                print("Summary writer not available at the moment")

    def _output_costs(self, epoch, data, mode):

        if self.verbose >= 1:
            if epoch == 0:
                print "Average " + mode + " cost before training:"
            else:
                print "Average " + mode + " cost at step %s:" % (epoch,)

        self._run_error_and_summaries(epoch, data, self.tf_summary_writer_train if mode == 'training'
                                                        else self.tf_summary_writer_validation)

    def store_encodings(self, data, name, graph=None):

        """ Encode the data and store the encoded versions on disk.
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :param graph: tf graph object, optional
        :return: self
        """

        filepath = os.path.join(self.encodings_path, name + '_encoded.npy')
        data.startStoring(filepath, self.num_hidden, np.float32)

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                feed_dict = self._create_feed_dict(data, self.batch_size)
                while feed_dict is not None:
                    # Encode the current batch
                    data.store(self.tf_session.run(self.encode, feed_dict=feed_dict))
                    feed_dict = self._create_feed_dict(data, self.batch_size)

        data.finishStoring(filepath)

    def build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        :return: self
        """

        self._create_placeholders()
        self._create_variables()

        hidden_probabilities_0 = self._sample_hidden_from_visible(self.input_data)
        self.encode = self._sample_states(hidden_probabilities_0)
        self.reconstruction = self._sample_visible_from_hidden(self.encode)
        positive_associations = tf.matmul(tf.transpose(self.input_data), hidden_probabilities_0)

        # Run the gibbs sampling steps except the final one, using the visible reconstruction as the starting point
        visible_probabilities_final = self.reconstruction
        for step in range(self.gibbs_sampling_steps-1):
            hidden_probabilities_final = self._sample_states(self._sample_hidden_from_visible(visible_probabilities_final))
            visible_probabilities_final = self._sample_visible_from_hidden(hidden_probabilities_final)

        # The final gibbs step uses the probabilities to avoid sampling noise
        hidden_probabilities_final = self._sample_hidden_from_visible(visible_probabilities_final)
        visible_probabilities_final = self._sample_visible_from_hidden(hidden_probabilities_final)

        negative_associations = tf.matmul(tf.transpose(visible_probabilities_final), hidden_probabilities_final)

        self.w_upd8 = self.W.assign_add(self.learning_rate *
                                        (positive_associations - negative_associations) / self.batch_size)
        self.bh_upd8 = self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hidden_probabilities_0 - hidden_probabilities_final, 0))
        self.bv_upd8 = self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - visible_probabilities_final, 0))

        variables = [self.W, self.bh_, self.bv_]

        regterm = self.compute_regularization(variables)

        self._create_cost_function_node(self.reconstruction, self.input_data, regterm=regterm)

    def compute_regularization(self, variables):
        """ Compute the regularization tensor.
        :param variables: list of model variables
        :return:
        """

        regularization = tf.constant(0.0)

        if self.regtype == 'l2':
            for v in variables:
                regularization = tf.add(regularization, tf.nn.l2_loss(v))

            regularization = tf.mul(self.l2reg, regularization)
        elif self.regtype == 'l1':
            for v in variables:
                regularization = tf.add(regularization, tf.reduce_sum(tf.abs(v)))

        return regularization

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, self.num_visible], name='x-input')

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        self.W = tf.Variable(tf.truncated_normal(shape=[self.num_visible, self.num_hidden], stddev=0.01), name='weights')
        self.bh_ = tf.Variable(tf.constant(0.0, shape=[self.num_hidden]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.constant(0.0, shape=[self.num_visible]), name='visible-bias')

    def _create_cost_function_node(self, data_reconstruction, data_actual, regterm=0):

        """ Create the cost function node.
        :param data_reconstruction: data as reconstructed by the model after encoding actual_data
        :param data_actual: the placeholder node for the actual activations of the visible units (pixel data)
        :param regterm: regularization term, currently only used in cost calculation
        :return: self
        """

        cost_cross_entropy = - tf.reduce_mean(data_actual * tf.log(tf.clip_by_value(data_reconstruction, 1e-10, float('inf'))) +
                                        (1 - data_actual) * tf.log(tf.clip_by_value(1 - data_reconstruction, 1e-10, float('inf'))))

        cost_mean_squared = tf.reduce_mean(tf.square(data_actual - data_reconstruction))

        with tf.name_scope("cost"):
            self.cost_cross_entropy = cost_cross_entropy + regterm
            self.cost_mean_squared = cost_mean_squared + regterm

    def _sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.

        :param visible: activations of the visible units
        :return: hidden probabilities
        """

        hidden_activations = tf.matmul(visible, self.W) + self.bh_

        if self.hidden_unit_type == 'logistic' or self.hidden_unit_type == 'bin':
            hidden_probabilities = tf.nn.sigmoid(hidden_activations)
        elif self.hidden_unit_type == 'gauss':
            hidden_probabilities = hidden_activations
        elif self.hidden_unit_type == 'relu':
            hidden_probabilities = tf.nn.relu(hidden_activations)
        else:
            hidden_probabilities = None

        return hidden_probabilities

    def _sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'logistic' or self.visible_unit_type == 'bin':
            visible_probabilities = tf.nn.sigmoid(visible_activation)
        elif self.visible_unit_type == 'gauss':
            visible_probabilities = visible_activation
        elif self.visible_unit_type == 'relu':
            visible_probabilities = tf.nn.relu(visible_activation)
        else:
            visible_probabilities = None

        return visible_probabilities

    def _sample_states(self, probabilities):
        """ Takes a tensor of probabilities and generates the actual states

        :param probabilities: tensor of probabilities
        :return : binary states
        """

        if self.hidden_unit_type == 'bin':
            states = tf.maximum(0.0, tf.sign(probabilities -
                                             tf.random_uniform(tf.shape(probabilities), minval=0.0, maxval=1.0,
                                                                 dtype=tf.float32, name="sample_rand"),
                                                                                                name="sample_binary"))
        elif self.hidden_unit_type == 'gauss':
            states = tf.truncated_normal(tf.shape(probabilities), mean=probabilities, stddev=self.stddev)
        else:
            states = probabilities

        return states

    def load_model(self, graph=None):

        """ Load a trained model from disk. The shape of the model
        (num_visible, num_hidden) and the number of gibbs sampling steps
        must be known in order to restore the model.
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.build_model()
                self._initialize_tf_utilities_and_ops(restore_previous_model=True)
                self.tf_saver.restore(self.tf_session, self.model_path)

    def _store_hidden_probabilities(self, epoch, data, numExamples):
        filepath = os.path.join(self.debug_path, 'hidden_probabilities_%03d.npy' % epoch)
        data.startStoring(filepath, self.num_hidden, np.float32, numRows = self.batch_size)

        feed_dict = self._create_feed_dict(data, self.batch_size)
        if feed_dict is not None:
            # Retrieve and store the hidden probabilities
            probs = self.tf_session.run(self.encode, feed_dict=feed_dict)
            data.store(probs)

        data.finishStoring()

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
