import tensorflow as tf
import numpy as np
import os
from os.path import expanduser

from yadlt.models.rbm_models import rbm_sql
from yadlt.utils import utilities

__author__ = 'Annika Lindh, Gabriele Angeletti'


class UnsupervisedDeepBeliefNetwork:

    """ Implementation of Deep Belief Network for Unsupervised Feature Extraction using TensorFlow.
    Based on the DeepBeliefNetwork implementation by blackecho.
    Data is fed from a DB for better memory management.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_visible, rbm_layers, visible_unit_type='gauss', model_name='puredbn', verbose=1,
                 rbm_gibbs_k=[1], rbm_learning_rates=[0.01], rbm_batch_sizes=[10], rbm_num_epochs=[10], rbm_stddev=0.1,
                 loss_func='mean_squared', regtype='none', l2reg=5e-4,
                 main_dir='puredbn/', models_dir='models/', data_dir='data/', summary_dir='logs/'):

        """
        :param num_visible: number of visible units
        :param rbm_layers: list containing the hidden units for each layer
        :param visible_unit_type: type of the visible units, 'bin' (binary) or 'gauss' (gaussian), default gauss
        :param model_name: name of the model, used as filename. string, default 'puredbn'
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 1
        :param rbm_gibbs_k: optional, number of gibbs sampling steps per layer, default 1 for all layers
        :param learning_rate: Initial learning rate
        :param batch_size: Size of the mini-batches for each RBM-layer
        :param num_epochs: Number of epochs for each RBM-layer
        :param rbm_stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param loss_function: Loss function. ['mean_squared', 'cross_entropy']
        :param regtype: regularization type
        :param l2reg: regularization parameter
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param models_dir: directory to store trained models
        :param data_dir: directory to store generated data
        :param summary_dir: directory to store tensorflow logs
        """

        # Setup the directories
        home = os.path.join(expanduser("~"), '.yadlt')
        self.model_name = model_name
        self.main_dir = os.path.join(home, main_dir)
        self.models_dir = os.path.join(home, models_dir)
        self.data_dir = os.path.join(home, data_dir)
        self.tf_summary_dir = os.path.join(home, summary_dir)
        self.model_path = os.path.join(self.models_dir, self.model_name)

        print('Creating %s directory to save/restore models' % (self.models_dir))
        utilities.create_dir(self.models_dir)
        print('Creating %s directory to save model generated data' % (self.data_dir))
        utilities.create_dir(self.data_dir)
        print('Creating %s directory to save tensorboard logs' % (self.tf_summary_dir))
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
                visible_unit_type = 'gauss'

            self.rbms.append(rbm_sql.RBMSQL(num_visible, num_hidden, visible_unit_type=visible_unit_type,
                model_name=self.model_name + '-' + rbm_str, verbose=self.verbose,
                gibbs_sampling_steps=rbm_params['gibbs_k'][i], learning_rate=rbm_params['learning_rate'][i],
                batch_size=rbm_params['batch_size'][i], num_epochs=rbm_params['num_epochs'][i], stddev=self.rbm_stddev,
                loss_func=self.loss_func, regtype=self.regtype, l2reg=self.l2reg,
                main_dir = self.main_dir, models_dir=os.path.join(self.models_dir, rbm_str),
                data_dir=os.path.join(self.data_dir, rbm_str), summary_dir=os.path.join(self.tf_summary_dir, rbm_str)))

            self.rbm_graphs.append(tf.Graph())

            # The hidden units from this layer are the visible units for the next layer
            num_visible = num_hidden

    def fit(self, conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_fetch_train_first, stmt_fetch_train, stmt_store_train,
            stmt_fetch_validation_first=None, stmt_fetch_validation=None, stmt_store_validation=None,
            restore_previous_model=False, graph=None):

        """ Fit the model to the data
        This is what should be called from outside and wrap around the internal training
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_shuffle: Statement to generate random numbers used for batch assignment
        :param stmt_fetch_train_first: Prepared statement for selecting the original data from training table
        :param stmt_fetch_train: Prepared statement for selecting the previously encoded data from training table
        :param stmt_store_train: Prepared statement for storing the encoded data in the training table
        :param stmt_fetch_train_first: optional, Prepared statement for selecting the original data from validation table
        :param stmt_fetch_validation: optional, Prepared statement for selecting the previously encoded data from validation table
        :param stmt_store_validation: optional Prepared statement for storing the encoded data in the validation table
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: TensorFlow graph object, optional
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self._train_model(conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_fetch_train_first, stmt_fetch_train, stmt_store_train,
                                  stmt_fetch_validation_first, stmt_fetch_validation, stmt_store_validation, restore_previous_model, graph)

    def _train_model(self, conSelect, conInsert, curSelect, curInsert, stmt_shuffle, stmt_fetch_train_first, stmt_fetch_train, stmt_store_train,
                     stmt_fetch_validation_first=None, stmt_fetch_validation=None, stmt_store_validation=None, restore_previous_model=False, graph=None):

        """ Train the model on the data
        :param conSelect: MySQL connection for fetching data (needs to be different from the insert connection to allow
                            concurrent batch-fetching and inserting)
        :param conInsert: MySQL connection for inserts and updates
        :param curSelect: Prepared MySQL cursor for fetching data
        :param curInsert: Prepared MySQL cursor for inserts and updates
        :param stmt_shuffle: Statement to generate random numbers used for batch assignment
        :param stmt_fetch_train_first: Prepared statement for selecting the original data from training table
        :param stmt_fetch_train: Prepared statement for selecting the previously encoded data from training table
        :param stmt_store_train: Prepared statement for storing the encoded data in the training table
        :param stmt_fetch_train_first: optional, Prepared statement for selecting the original data from validation table
        :param stmt_fetch_validation: optional, Prepared statement for selecting the previously encoded data from validation table
        :param stmt_store_validation: optional Prepared statement for storing the encoded data in the validation table
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

          # The first layer fetches from the original data
          if i == 0:
              current_fetch_train = stmt_fetch_train_first
              current_fetch_validation = stmt_fetch_validation_first
          else:
              current_fetch_train = stmt_fetch_train
              current_fetch_validation = stmt_fetch_validation

          # Fully train this RBM-layer
          self.rbms[i].fit(conSelect, conInsert, curSelect, curInsert, stmt_shuffle, current_fetch_train, current_fetch_validation, restore_previous_model, graph)

          # Generate encodings for the full dataset (training and validation sets) and store to DB
          self.rbms[i].store_encodings(conSelect, conInsert, curSelect, curInsert, current_fetch_train, stmt_store_train, graph)
          if (current_fetch_validation is not None) and (stmt_store_validation is not None):
              self.rbms[i].store_encodings(conSelect, conInsert, curSelect, curInsert, current_fetch_validation, stmt_store_validation, graph)

    def export_filters(self, img_shape, num_imgs, padding):
        """ Export the learned filters as images of the same shape as the input data. """

        imgs = None
        iLayer = 0
        for currentRBM in self.rbms:
            if iLayer == 0:
                imgs = utilities.reshape_weights_to_images(
                    np.transpose(currentRBM.get_model_parameters(restore_previous_model=True)['W']),
                    img_shape, num_imgs)
            else:
                imgs = utilities.visualize_deep_filters(
                    np.transpose(currentRBM.get_model_parameters(restore_previous_model=True)['W']), imgs)

            utilities.save_tiled_images(self.model_path + self.model_name + "_layer" + str(iLayer) + ".png", imgs, img_shape, padding)
            iLayer += 1

