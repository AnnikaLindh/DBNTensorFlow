import os
from os.path import expanduser
import tensorflow as tf
import numpy as np

from yadlt.utils import utilities

__author__ = 'Annika Lindh'


class DBNFinetuned:

    """ The full DBN constructed from a pre-trained stack of RBMs.
    This implementation assumes at least 2 RBMs. If there's only a single RBM there's no need for this stage.
    The interface of the class is sklearn-like.
    """

    def __init__(self, srbm_parameters=None, layers=None,
                 visible_unit_type='logistic', hidden_unit_type='bin',
                 model_name='dbn_finetuned', verbose=1, gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10,
                 num_epochs=10, stddev=0.01, loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 main_dir='dbn_finetuned/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 encodings_dir='encodings/'):

        """
        :param srbm_parameters: the parameters from the stack of RBMs that will be finetuned
        :param layers: the amount of units of each layer, used instead of srbm_parameters when loading
        :param visible_unit_type: type of the visible units, logistic, bin, gauss or relu
        :param hidden_unit_type: type of the hidden units, logistic, bin, gauss or relu
        :param model_name: name of the model, used as filename. string, default 'rbm'
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
        :param encodings_dir: directory to store final encoded versions of the data
        """

        # Depending on whether the model is being loaded or created, the first or the second of these will be used
        self.srbm_parameters = srbm_parameters
        self.layers = layers
        self.encoded_size = len(srbm_parameters[-1]['bh_']) if srbm_parameters is not None else layers[-1]
        self.num_layers = len(srbm_parameters) if srbm_parameters is not None else len(layers)

        self.visible_unit_type = visible_unit_type
        self.hidden_unit_type = hidden_unit_type

        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev

        self.loss_func = loss_func
        self.regtype = regtype
        self.l2reg = l2reg

        self.verbose = verbose
        home = os.path.join(expanduser("~"), '.yadlt')
        self.main_dir = os.path.join(home, main_dir)
        self.models_dir = os.path.join(home, models_dir)
        self.data_dir = os.path.join(home, data_dir)
        self.tf_summary_dir = os.path.join(home, summary_dir)
        self.model_path = os.path.join(self.models_dir, model_name)
        self.encodings_path = os.path.join(self.models_dir, encodings_dir)

        utilities.create_dir(self.models_dir)
        utilities.create_dir(self.data_dir)
        utilities.create_dir(self.encodings_path)
        utilities.create_dir(self.tf_summary_dir)

        # tensorflow nodes
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer_train = None
        self.tf_summary_writer_validation = None
        self.encoding_weights = list()
        self.encoding_biases = list()
        self.decoding_weights = list()
        self.decoding_biases = list()
        self.input_data = None
        self.encoded_data = None
        self.decoded_data = None
        self.updated_encoding_weights = None
        self.updated_encoding_biases = None
        self.updated_decoding_weights = None
        self.updated_decoding_biases = None
        self.cost_components = None
        self.cost_wake = None
        self.cost_sleep = None
        self.generative_samples = None

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
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

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

            if i % 10 == 0:
                if self.verbose >= 2:
                    print "Running error calculations."

                self._output_costs(i, data_training, 'training')
                if data_validation is not None:
                    self._output_costs(i, data_validation, 'validation')

            if stopEarly:
                break

    def _run_train_step(self, data):

        """ Run a training step. A training step is made by randomly shuffling the training set and then training
        the model on one mini-batch at a time.
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

        feed_dict = self._create_feed_dict(data, self.batch_size)
        while feed_dict is not None:
            updated_values = self.tf_session.run(self.updated_encoding_weights + self.updated_encoding_biases +
                       self.updated_decoding_weights + self.updated_decoding_biases, feed_dict=feed_dict)

            # Check for NaN values due to exploding weights (if this happens, try lowering the learning rate)
            hasNan = False
            for iValue in range(0, len(updated_values)):
                if np.isfinite(updated_values[iValue]).all() == False:
                    hasNan = True
                    print "WARNING: %s %s have NaN values in layer %s." % (
                                                        "Encoding" if (iValue/self.num_layers) < 2 else "Decoding",
                                                        "weights" if (iValue/self.num_layers) % 2 == 0 else "biases",
                                                        iValue % self.num_layers,)
            if hasNan:
                feed_dict = None
                return True
            else:
                feed_dict = self._create_feed_dict(data, self.batch_size)

        return False

    def _create_feed_dict(self, datahandler, batchSize):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
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
            cost_updates.append(self.cost_cross_entropy_wake)
            cost_updates.append(self.cost_cross_entropy_sleep)
            cost_names.append('cross_entropy')
        if self.loss_func == 'mean_squared' or self.loss_func == 'all':
            cost_updates.append(self.cost_mean_squared_wake)
            cost_updates.append(self.cost_mean_squared_sleep)
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

        for i in range(0, len(cost_names)):
            avgCost_wake = totalBatchCosts[i*2] / numBatches
            avgCost_sleep = totalBatchCosts[i*2+1] / numBatches
            avgCost_total = avgCost_wake + avgCost_sleep
            if self.verbose >= 1:
                print "\t" + cost_names[i] + '_wake: ' + str(avgCost_wake)
                print "\t" + cost_names[i] + '_sleep: ' + str(avgCost_sleep)
                print "\t" + cost_names[i] + '_total: ' + str(avgCost_total)
            try:
                summary_str = tf.Summary(value=[tf.Summary.Value(tag=cost_names[i] + '_wake', simple_value=avgCost_wake)])
                writer.add_summary(summary_str, epoch)
                summary_str = tf.Summary(value=[tf.Summary.Value(tag=cost_names[i] + '_sleep', simple_value=avgCost_sleep)])
                writer.add_summary(summary_str, epoch)
                summary_str = tf.Summary(value=[tf.Summary.Value(tag=cost_names[i] + '_total', simple_value=avgCost_total)])
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
        data.startStoring(filepath, self.encoded_size, np.float32)

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                feed_dict = self._create_feed_dict(data, self.batch_size)
                while feed_dict is not None:
                    # Encode the current batch
                    data.store(self.tf_session.run(self.encoded_data, feed_dict=feed_dict))
                    feed_dict = self._create_feed_dict(data, self.batch_size)

        data.finishStoring(filepath)

    def build_model(self):

        """ Build the DBN model in TensorFlow.
        :return: self
        """

        self._create_placeholders()
        self._create_variables()

        # Prepare the encoding and decoding functionality from a user-supplied input
        self.encoded_data = self._encode_or_decode(self.input_data, "encode", stochastic=False)['states'][-1]
        self.decoded_data = self._encode_or_decode(self.input_data, "decode", stochastic=False)['states'][-1]

        self._create_learning_rules()
        self._create_cost_function_nodes()

        self._create_generative_sampling_nodes()

    def _calculate_updates(self, weights, biases, true_states, predicted_activations):
        new_weights = list()
        new_biases = list()

        for iWeightLayer in range(0, len(weights)):
            # The difference between the "true" states and the reconstructed states is used in both updates below
            reconstruction_difference = true_states[-(iWeightLayer + 2)] - predicted_activations[iWeightLayer]

            # Update the weights based on the difference in associations between the "true" states and the next layer
            # and the associations between the predicted states and the next layer, averaged over the batch size
            new_weights.append(weights[iWeightLayer].assign_add((self.learning_rate/float(self.batch_size)) *
                                tf.matmul(tf.transpose(true_states[-(iWeightLayer + 1)]), reconstruction_difference)))

            # Update the biases (using reduce_mean so it's already averaged over the batch size)
            new_biases.append(biases[iWeightLayer].assign_add(self.learning_rate *
                                                              tf.reduce_mean(reconstruction_difference, 0)))

        return {'new_weights': new_weights, 'new_biases': new_biases}

    def _create_learning_rules(self):
        """
        This function implements the contrastive wake-sleep algorithm for unsupervised learning.
        The implementation is based on (Hinton et al., 2006) and (Hinton et al., 1995).
        :return:
        """

        # Get wake-phase statistics
        wake_encoded_states = self._encode_or_decode(self.input_data, mode="encode", stochastic=True)['states']
        wake_decoded_activations = \
            self._encode_or_decode(wake_encoded_states[-1], mode="decode", stochastic=True)['activations']

        # Calculate the weight updates for the decoding (generative) weights
        wake_encoded_states = [self.input_data] + wake_encoded_states
        updated_parameters = self._calculate_updates(weights=self.decoding_weights,
                                                          biases=self.decoding_biases,
                                                          true_states=wake_encoded_states,
                                                          predicted_activations=wake_decoded_activations)
        self.updated_decoding_weights = updated_parameters['new_weights']
        self.updated_decoding_biases = updated_parameters['new_biases']

        # Perform gibbs sampling over the top two layers, initialized by the training data (which we already encoded)
        topHiddenStates = wake_encoded_states[-1]
        for step in range(self.gibbs_sampling_steps):
            topVisibleStates = self._sample_states(
                self._calculate_activations(tf.matmul(topHiddenStates, self.decoding_weights[0]) + self.decoding_biases[0],
                                            self.hidden_unit_type), self.hidden_unit_type)
            topHiddenStates = self._sample_states(
                self._calculate_activations(tf.matmul(topVisibleStates, self.encoding_weights[-1]) + self.encoding_biases[-1],
                                            self.hidden_unit_type), self.hidden_unit_type)


        # Get sleep-phase statistics, initializing with the top hidden states after gibbs sampling
        sleep_decoded_states = self._encode_or_decode(topHiddenStates, mode="decode", stochastic=True)['states']
        sleep_encoded_activations = \
            self._encode_or_decode(sleep_decoded_states[-1], mode="encode", stochastic=True)['activations']

        # Calculate the weight updates for the encoding (recognition) weights
        sleep_decoded_states = [topHiddenStates] + sleep_decoded_states
        updated_parameters = self._calculate_updates(weights=self.encoding_weights,
                                                          biases=self.encoding_biases,
                                                          true_states=sleep_decoded_states,
                                                          predicted_activations=sleep_encoded_activations)
        self.updated_encoding_weights = updated_parameters['new_weights']
        self.updated_encoding_biases = updated_parameters['new_biases']

        # Store values for cost calculation
        self.cost_components = {'visible_actual': self.input_data,
                                'visible_reconstructed': wake_decoded_activations[-1],
                                'hidden_actual': topHiddenStates,
                                'hidden_reconstructed': sleep_encoded_activations[-1]}

    def _encode_or_decode(self, input_data, mode, stochastic=True):
        """ This function either goes through the encoding half or the decoding half of the net, based on the mode.
        :param input_data: The data to encode or decode (based on mode)
        :param mode: encode or decode
        :return: A dictionary with two lists: states and activations
        """

        if mode == "encode":
            weights = self.encoding_weights
            biases = self.encoding_biases
        elif mode == "decode":
            weights = self.decoding_weights
            biases = self.decoding_biases
        else:
            raise ValueError("Argument 'mode' must be either 'encode' or 'decode'.")

        unit_type = self.hidden_unit_type

        activations = list()
        states = list()
        currentData = input_data
        numLayers = len(weights)
        for iLayer in range(0, numLayers):
            # The final layer of the decoding pass should use the visible unit type
            if (iLayer == (numLayers - 1)) and mode == "decode":
                unit_type = self.visible_unit_type

            currentActivations = self._calculate_activations(tf.matmul(currentData, weights[iLayer]) + biases[iLayer], unit_type)
            activations.append(currentActivations)
            currentData = self._sample_states(currentActivations, unit_type, stochastic=stochastic)
            states.append(currentData)

        return {'states': states, 'activations': activations}

    def _calculate_activations(self, raw_output, unit_type):
        """ Takes a tensor of activations and generates the actual states

        :param raw_output: raw output from the unit
        :param unit_type: determines the activation function
        :return : tensor of activation values
        """

        if unit_type == 'bin' or unit_type == 'logistic':
            activations = tf.nn.sigmoid(raw_output)
        elif unit_type == 'relu':
            activations = tf.nn.relu(raw_output)
        else:
            activations = raw_output

        return activations

    def _sample_states(self, activations, unit_type, stochastic=True):
        """ Takes a tensor of activations and generates the actual states

        :param activations: tensor of activations
        :param unit_type: determines the sampling type
        :param stochastic: default True, if stochastic, then an element of randomness exists in the sampling,
                            otherwise, deterministic sampling is performed
        :return : unit states
        """

        if unit_type == 'bin':
            if stochastic:
                states = tf.maximum(0.0,
                                    tf.sign(activations - tf.random_uniform(tf.shape(activations), minval=0.0,
                                                        maxval=1.0, dtype=tf.float32, name="sample_rand"),
                                                                                            name="sample_binary"))
            else:
                states = tf.round(activations, name="round_binary")

        elif unit_type == 'gauss':
            states = tf.truncated_normal(tf.shape(activations), mean=activations,
                                         stddev=self.stddev if stochastic else 0.0)

        else:
            states = activations

        return states

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        # The shape of the data differs when encoding and decoding
        self.input_data = tf.placeholder('float', [None, None], name='x-input')

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        if self.srbm_parameters == None:
            # Build the encoding half of the DBN
            for i in range(0, self.num_layers-1):
                with tf.name_scope('encoding_layer_' + str(i)) as scope:
                    self.encoding_weights.append(tf.Variable(tf.zeros(shape=[self.layers[i], self.layers[i+1]]), name='weights'))
                    self.encoding_biases.append(tf.Variable(tf.zeros(shape=[self.layers[i+1]]), name='biases'))

            # Build the decoding part of the DBN
            iLayerNum = 0
            for i in range(self.num_layers-1, 0, -1):
                with tf.name_scope('decoding_layer_' + str(iLayerNum)) as scope:
                    self.decoding_weights.append(tf.Variable(tf.zeros(shape=[self.layers[i], self.layers[i-1]]), name='weights'))
                    self.decoding_biases.append(tf.Variable(tf.zeros(shape=[self.layers[i-1]]), name='biases'))
                    iLayerNum += 1

        else:
            # Build the encoding half of the DBN
            for i in range(0, self.num_layers):
                with tf.name_scope('encoding_layer_' + str(i)) as scope:
                    self.encoding_weights.append(tf.Variable(self.srbm_parameters[i]['W'], name='weights'))
                    self.encoding_biases.append(tf.Variable(self.srbm_parameters[i]['bh_'], name='biases'))

            # Build the decoding part of the DBN
            iLayerNum = 0
            for i in range(self.num_layers-1, -1, -1):
                with tf.name_scope('decoding_layer_' + str(iLayerNum)) as scope:
                    self.decoding_weights.append(tf.Variable(tf.transpose(self.srbm_parameters[i]['W']), name='weights'))
                    self.decoding_biases.append(tf.Variable(self.srbm_parameters[i]['bv_'], name='biases'))
                    iLayerNum += 1

    def _create_cost_function_nodes(self):

        """ Create the cost function node.
        :return: self
        """

        with tf.name_scope("cost"):
            cost_cross_entropy_wake = -tf.reduce_mean(
                self.cost_components['visible_actual'] * tf.log(
                    tf.clip_by_value(self.cost_components['visible_reconstructed'], 1e-10, float('inf'))) +
                    (1 - self.cost_components['visible_actual']) * tf.log(
                         tf.clip_by_value(1 - self.cost_components['visible_reconstructed'], 1e-10, float('inf'))))
            cost_cross_entropy_sleep = -tf.reduce_mean(
                self.cost_components['hidden_actual'] * tf.log(
                    tf.clip_by_value(self.cost_components['hidden_reconstructed'], 1e-10, float('inf'))) +
                    (1 - self.cost_components['hidden_actual']) * tf.log(
                         tf.clip_by_value(1 - self.cost_components['hidden_reconstructed'], 1e-10, float('inf'))))

            cost_mean_squared_wake = tf.reduce_mean(tf.square(
                self.cost_components['visible_actual'] - self.cost_components['visible_reconstructed']))
            cost_mean_squared_sleep = tf.reduce_mean(tf.square(
                self.cost_components['hidden_actual'] - self.cost_components['hidden_reconstructed']))

            reg_wake = self._calculate_regularization([self.updated_decoding_weights, self.updated_decoding_biases])
            reg_sleep = self._calculate_regularization([self.updated_encoding_weights, self.updated_encoding_biases])
            self.cost_cross_entropy_wake = cost_cross_entropy_wake + reg_wake
            self.cost_cross_entropy_sleep = cost_cross_entropy_sleep + reg_sleep
            self.cost_mean_squared_wake = cost_mean_squared_wake + reg_wake
            self.cost_mean_squared_sleep = cost_mean_squared_sleep + reg_sleep

    def _create_generative_sampling_nodes(self):
        # Start off the top hidden states from the data feed
        topHiddenStates = self._encode_or_decode(self.input_data, "encode", stochastic=True)['states'][-1]

        # Perform gibbs sampling over the top two layers, initialized by the input data (which we already encoded)
        for step in range(1000):
            topVisibleStates = self._sample_states(
                self._calculate_activations(tf.matmul(topHiddenStates, self.decoding_weights[0]) + self.decoding_biases[0],
                                            self.hidden_unit_type), self.hidden_unit_type)
            topHiddenStates = self._sample_states(
                self._calculate_activations(tf.matmul(topVisibleStates, self.encoding_weights[-1]) + self.encoding_biases[-1],
                                            self.hidden_unit_type), self.hidden_unit_type)

        self.generative_samples = self._encode_or_decode(topHiddenStates, "decode", stochastic=True)['states'][-1]

    def _calculate_regularization(self, parameters):
        regularization = tf.constant(0.0)

        if self.regtype == 'l2':
            for param in parameters:
                regularization = tf.add(regularization, tf.nn.l2_loss(param))

            regularization = tf.mul(self.l2reg, regularization)
        elif self.regtype == 'l1':
            for param in parameters:
                regularization = tf.add(regularization, tf.reduce_sum(tf.abs(param)))

        return regularization

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


    def export_features(self, img_shape, num_imgs, featuresPerRow, padding):
        """ Export the learned features as images of the same shape as the input data.
        :param img_shape: Shape of the feature images.
        :param num_imgs: Number of images per neuron.
        :param featuresPerRow: Numbers of features per row in the output image.
        :param padding: Amount of padding in pixels between the images.
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                imgs = None
                for iLayer in range(0, self.num_layers-1):
                    if iLayer == 0:
                        [imgs, imgs_norm] = utilities.reshape_weights_to_images(
                            np.transpose(self.encoding_weights[iLayer].eval()), img_shape, num_imgs)
                    else:
                        [imgs, imgs_norm] = utilities.visualize_deep_features(
                            np.transpose(self.encoding_weights[iLayer].eval()), imgs)

                    utilities.save_tiled_images(self.model_path + "_layer" + str(iLayer) + ".png",
                                                imgs, img_shape, featuresPerRow, padding)
                    utilities.save_tiled_images(self.model_path + "_layer" + str(iLayer) + "_norm.png",
                                                imgs_norm, img_shape, featuresPerRow, padding)

    def sample_from_generative(self, data_samples, num_samples, out_dir, visible_size):
        """
        Visualize what sort of samples the DBN "believes in".
        Start off with one mini-batch of samples, run extended Gibbs sampling, export the current visible states.
        Repeat with a large amount of Gibbs steps in between to get different types of samples.
        :return:
        """
        utilities.create_dir(out_dir)

        g = self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                # Generate the samples, starting from the sample set and then continuing from the last generated sample
                for iSample in range(0, num_samples):
                    filepath = os.path.join(out_dir, 'samples_' + str(iSample) + '.npy')
                    data_samples.startStoring(filepath, visible_size, np.float32)
                    current_samples = self.tf_session.run(self.generative_samples,
                                                          feed_dict=self._create_feed_dict(data_samples,
                                                                                           self.batch_size))
                    data_samples.store(current_samples)
                    data_samples.finishStoring(filepath)
