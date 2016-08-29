import tensorflow as tf
import numpy as np
import os
from os.path import expanduser

from yadlt.models.dbn_models.rbm import RBM
from yadlt.models.dbn_models.dbn_finetuned import DBNFinetuned
from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


class DeepBeliefNetwork:

    """ Implementation of Deep Belief Network for Unsupervised Feature Extraction using TensorFlow.
    Based on the DeepBeliefNetwork implementation from blackecho's Deep-Learning-Tensorflow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_visible, rbm_layers, visible_unit_type='logistic', hidden_unit_type='bin',
                 model_name='dbn', verbose=1,
                 rbm_gibbs_k=[1], rbm_learning_rates=[0.01], rbm_batch_sizes=[10], rbm_num_epochs=[10], rbm_stddev=0.01,
                 loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 main_dir='dbn/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 rbm_debug_dir='debug/', debug_hidden_units=0):

        """
        :param num_visible: number of visible units
        :param rbm_layers: list containing the hidden units for each layer
        :param visible_unit_type: type of the visible units, logistic, bin, gauss or relu
        :param hidden_unit_type: type of the hidden units, logistic, bin, gauss or relu
        :param model_name: name of the model, used as filename. string, default 'dbn'
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 1
        :param rbm_gibbs_k: optional, number of gibbs sampling steps per layer, default 1 for all layers
        :param learning_rate: Initial learning rate
        :param batch_size: Size of the mini-batches for each RBM-layer
        :param num_epochs: Number of epochs for each RBM-layer
        :param rbm_stddev: optional, default 0.01. Ignored if visible_unit_type is not 'gauss'
        :param loss_function: Loss function. ['mean_squared', 'cross_entropy', 'all']
        :param regtype: regularization type
        :param l2reg: regularization parameter
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param models_dir: directory to store trained models
        :param data_dir: directory to store generated data
        :param summary_dir: directory to store tensorflow logs
        :param rbm_debug_dir: directory to store debug data for the RBMs
        :param debug_hidden_units: number of neurons for which to visualise hidden probabilities
        """

        # Setup the directories
        home = os.path.join(expanduser("~"), '.yadlt')
        self.model_name = model_name
        self.main_dir = os.path.join(home, main_dir)
        self.models_dir = os.path.join(home, models_dir)
        self.data_dir = os.path.join(home, data_dir)
        self.tf_summary_dir = os.path.join(home, summary_dir)
        self.model_path = os.path.join(self.models_dir, self.model_name)
        self.rbm_debug_dir = rbm_debug_dir
        self.rbm_debug_hidden_units = debug_hidden_units

        utilities.create_dir(self.models_dir)
        utilities.create_dir(self.data_dir)
        utilities.create_dir(self.tf_summary_dir)

        self.verbose = verbose

        # TensorFlow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None

        # Setup the RBM-layers
        self.num_visible = num_visible
        self.rbm_layers = rbm_layers
        self.rbm_num_epochs = rbm_num_epochs
        self.rbm_gibbs_k = rbm_gibbs_k
        self.rbm_batch_sizes = rbm_batch_sizes
        self.rbm_learning_rates = rbm_learning_rates
        self.visible_unit_type = visible_unit_type
        self.hidden_unit_type = hidden_unit_type
        self.rbm_stddev = rbm_stddev
        self.loss_func = loss_func
        self.regtype = regtype
        self.l2reg = l2reg
        self._setup_rbm_layers()

    def _setup_rbm_layers(self):

        # Fill up the params where only a default is specified
        rbm_params = {'num_epochs': self.rbm_num_epochs, 'gibbs_k': self.rbm_gibbs_k, 'batch_size': self.rbm_batch_sizes,
                      'learning_rate': self.rbm_learning_rates}
        for p in rbm_params:  # cycles through the key-names
            if len(rbm_params[p]) != len(self.rbm_layers):
                # The current parameter is not specified by the user, should default it for all the layers
                rbm_params[p] = [rbm_params[p][0] for _ in self.rbm_layers]  # repeats the single value

        # Generate the RBM instances that make up the DBN layers
        self.rbms = []
        self.rbm_graphs = []
        num_visible = self.num_visible
        for i, num_hidden in enumerate(self.rbm_layers):
            rbm_str = 'rbmlayer-' + str(i + 1)
            if i == 0:
                visible_unit_type = self.visible_unit_type
            else:
                visible_unit_type = self.hidden_unit_type

            self.rbms.append(RBM(
                num_visible, num_hidden, visible_unit_type=visible_unit_type, hidden_unit_type=self.hidden_unit_type,
                model_name=self.model_name + '-' + rbm_str, verbose=self.verbose,
                gibbs_sampling_steps=rbm_params['gibbs_k'][i], learning_rate=rbm_params['learning_rate'][i],
                batch_size=rbm_params['batch_size'][i], num_epochs=rbm_params['num_epochs'][i], stddev=self.rbm_stddev,
                loss_func=self.loss_func, regtype=self.regtype, l2reg=self.l2reg,
                main_dir = self.main_dir, models_dir=os.path.join(self.models_dir, rbm_str),
                data_dir=os.path.join(self.data_dir, rbm_str), summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                debug_dir=self.rbm_debug_dir, debug_hidden_units=self.rbm_debug_hidden_units))

            self.rbm_graphs.append(tf.Graph())

            # The hidden units from this layer are the visible units for the next layer
            num_visible = num_hidden

    def fit(self, data_training, data_validation=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data
        This is what should be called from outside and wrap around the internal training
        :param data_training: Object that provides mini-batches for feeding and a function to save the encoded version
        :param data_validation: Object that provides mini-batches for feeding and a function to save the encoded version
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: TensorFlow graph object, optional
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self._train_model(data_training, data_validation, restore_previous_model, graph)

    def _train_model(self, data_training, data_validation=None, restore_previous_model=False, graph=None):

        """ Train the model on the data
        :param data_training: Object that provides mini-batches for feeding and a function to save the encoded version
        :param data_validation: Object that provides mini-batches for feeding and a function to save the encoded version
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: TensorFlow graph object, optional
        :return: self
        """

        # TODO Supply human-validation image(s) whose reconstruction is saved after each layer's training

        # Train each layer-RBM with the outputs from one being the inputs for the next
        nLayers = len(self.rbms)
        for i in range(0, nLayers):
          print('Training layer {} of {}...'.format(i + 1, nLayers))

          # Fully train this RBM-layer
          self.rbms[i].fit(data_training, data_validation, restore_previous_model, graph)

          # Generate and store encodings for the training and validation sets
          self.rbms[i].store_encodings(data_training, 'training', graph)
          self.rbms[i].store_encodings(data_validation, 'validation', graph)

    def get_finetuning_DBN(self, model_name, main_dir, verbose=1, gibbs_sampling_steps=1, learning_rate=0.01,
                 batch_size=10, num_epochs=10, stddev=0.0, loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 models_dir='models/', data_dir='data/', summary_dir='logs/',encodings_dir='encodings/',
                 restore_previous_model=True):
        """ Perform unsupervised finetuning on the full DBN built from the pre-trained stack of RBMs. """

        srbmParameters = list()
        for currentRBM in self.rbms:
            srbmParameters.append(currentRBM.get_model_parameters(restore_previous_model=restore_previous_model))

        finetuning_dbn = DBNFinetuned(srbmParameters,
                 visible_unit_type=self.visible_unit_type, hidden_unit_type=self.visible_unit_type,
                 model_name=model_name, verbose=verbose, gibbs_sampling_steps=gibbs_sampling_steps,
                 learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, stddev=stddev,
                 loss_func=loss_func, regtype=regtype, l2reg=l2reg,
                 main_dir=main_dir, models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
                 encodings_dir=encodings_dir)

        return finetuning_dbn

    def export_features(self, img_shape, num_imgs, featuresPerRow, padding):
        """ Export the learned features as images of the same shape as the input data.
        :param img_shape: Shape of the feature images.
        :param num_imgs: Number of images per neuron.
        :param featuresPerRow: Numbers of features per row in the output image.
        :param padding: Amount of padding in pixels between the images.
        """

        imgs = None
        iLayer = 0
        for currentRBM in self.rbms:
            if iLayer == 0:
                [imgs, imgs_norm] = utilities.reshape_weights_to_images(
                    np.transpose(currentRBM.get_model_parameters(restore_previous_model=True)['W']),
                    img_shape, num_imgs)
            else:
                [imgs, imgs_norm] = utilities.visualize_deep_features(
                    np.transpose(currentRBM.get_model_parameters(restore_previous_model=True)['W']), imgs)

            utilities.save_tiled_images(self.model_path + "_layer" + str(iLayer) + ".png",
                                        imgs, img_shape, featuresPerRow, padding)
            utilities.save_tiled_images(self.model_path + "_layer" + str(iLayer) + "_norm.png",
                                        imgs_norm, img_shape, featuresPerRow, padding)
            iLayer += 1

    def load_model(self):
        nLayers = len(self.rbms)

        for i in range(0, nLayers):
            print('Loading layer {} of {}...'.format(i + 1, nLayers))
            self.rbms[i].load_model(self.rbm_graphs[i])

    def store_encodings(self, data, name):
        """ Encode the data and store the encoded versions on disk.
        :param data: Object that provides mini-batches for feeding and a function to save the encoded version
        :return: self
        """

        nLayers = len(self.rbms)

        for i in range(0, nLayers):
            print('Encoding layer {} of {}...'.format(i + 1, nLayers))
            self.rbms[i].store_encodings(data, name, self.rbm_graphs[i])
