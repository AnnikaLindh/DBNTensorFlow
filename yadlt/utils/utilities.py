import os, errno
from scipy import misc
import tensorflow as tf
import numpy as np
from PIL import Image

__author__ = 'Gabriele Angeletti, Annika Lindh'


# ################### #
#   Network helpers   #
# ################### #


def xavier_init(fan_in, fan_out, const=1):
    """ Xavier initialization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


# ################ #
#   Data helpers   #
# ################ #


def gen_batches(data, batch_size):
    """ Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch

    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]


def conv2bin(data):
    """ Convert a matrix of probabilities into
    binary values. If the matrix has values <= 0 or >= 1, the values are
    normalized to be in [0, 1].

    :type data: numpy array
    :param data: input matrix

    :return: converted binary matrix
    """
    if data.min() < 0 or data.max() > 1:
        data = normalize(data)

    out_data = data.copy()

    for i, sample in enumerate(out_data):

        for j, val in enumerate(sample):

            if np.random.random() <= val:
                out_data[i][j] = 1
            else:
                out_data[i][j] = 0

    return out_data


def normalize(data):
    """ Normalize the data to be in the [0, 1] range.

    :param data:

    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


def bins2sets(bin_data):
    """ Convert a binary matrix into a collection of sets.
    This function is used to convert binary matrix of feature activations into sets
    representing which feature detector was activated for which input sample.
    For example the matrix [ [1, 0, 1, 0], [0, 0, 1, 0] ] will be converted in:
    [ ['f0', 'f2'], ['f2'] ]

    :type bin_data: numpy array
    :param bin_data: binary matrix

    :return: list of sets representing the binary matrix
    """

    feat = 'f'  # feature id
    out_data = {}

    for i, sample in enumerate(bin_data):
        sample_set = set()

        for j, activation in enumerate(sample):
            if activation == 1.:
                sample_set.add(feat + str(j))

        out_data[i] = sample_set

    return out_data


def masking_noise(data, sess, v):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param data: array_like, Input data
    :param sess: TensorFlow session
    :param v: fraction of elements to distort, float
    :return: transformed data
    """

    data_noise = data.copy()
    rand = tf.random_uniform(data.shape)
    data_noise[sess.run(tf.nn.relu(tf.sign(v - rand))).astype(np.bool)] = 0

    return data_noise


def salt_and_pepper_noise(X, v):
    """ Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.
    :param X: array_like, Input data
    :param v: int, fraction of elements to distort
    :return: transformed data
    """
    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise

# ############# #
#   Utilities   #
# ############# #


def expand_args(**args_to_expand):
    """Expands all the lists in args_to_expand into the length of layers.
    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE: the user can just specify one parameter and this function will expand it
    """
    layers = args_to_expand['layers']
    for key, val in args_to_expand.iteritems():
        if isinstance(val, list) and len(val) != len(layers):
            args_to_expand[key] = [val[0] for _ in layers]

    return args_to_expand


def flag_to_list(flagval, flagtype):

    if flagtype == 'int':
        return [int(_) for _ in flagval.split(',') if _]

    elif flagtype == 'float':
        return [float(_) for _ in flagval.split(',') if _]

    elif flagtype == 'str':
        return [_ for _ in flagval.split(',') if _]

    else:
        raise Exception("incorrect type")


def str2actfunc(act_func):
    if act_func == 'sigmoid':
        return tf.nn.sigmoid

    elif act_func == 'tanh':
        return tf.nn.tanh

    elif act_func == 'relu':
        return tf.nn.relu


def random_seed_np_tf(seed):
    """ seed numpy and tensorflow random number generators.
    :param seed: seed parameter
    """
    if seed >= 0:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        return True
    else:
        return False


def gen_image(img, width, height, outfile, img_type='grey'):
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))


def get_weights_as_images(self, weights, width, height, outdir='img/', n_images=10, img_type='grey'):
    """ Create and save the weights of the hidden units with respect to the
    visible units as images.
    :param weights:
    :param width:
    :param height:
    :param outdir:
    :param n_images:
    :param img_type:
    :return: self
    """

    perm = np.random.permutation(weights.shape[1])[:n_images]

    for p in perm:
        w = np.array([i[p] for i in weights])
        image_path = outdir + 'w_{}.png'.format(p)
        gen_image(w, width, height, image_path, img_type)

def create_dir(dirpath):
    """
    """

    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def reshape_weights_to_images(data, img_shape, num_imgs):

    """ data = transposed weights """

    img_size = img_shape[0]*img_shape[1]

    imgs = list()
    imgs_norm = list()
    for tile_row in range(0,data.shape[0]):
        img_row = list()
        img_row_norm = list()
        for tile_col in range(0,num_imgs):
            row = data[tile_row,tile_col*img_size:(tile_col+1)*img_size]
            img_row.append(row)
            img_row_norm.append((row-np.min(row))/(np.max(row)-np.min(row)) * 2.0 - 1.0)
        imgs.append(img_row)
        imgs_norm.append(img_row_norm)

    return [imgs, imgs_norm]

def visualize_deep_features(weights, images):
    assert (len(images) == weights.shape[1])

    numFeatures = weights.shape[0]

    imgs_out = list()
    imgs_out_norm = list()

    # Go through all the features of this weight-layer
    for i in range(0,numFeatures):
        img_row = list()
        img_row_norm = list()
        # Combine the previous layer's features and weight them
        rowMat = np.row_stack(images[0]) * weights[i,0]
        for j in range(1,weights.shape[1]):
            rowMat += np.row_stack(images[j]) * weights[i,j]
        # Re-normalize to a range between -1.0 and 1.0
        rowMat /= float(weights.shape[1])
        for row in rowMat:
            img_row.append(row)
            img_row_norm.append((row-np.min(row))/(np.max(row)-np.min(row)) * 2.0 - 1.0)
        imgs_out.append(img_row)
        imgs_out_norm.append(img_row_norm)

    return [imgs_out, imgs_out_norm]

def save_tiled_images(path, images, img_shape, featuresPerRow, padding):

    nImgRows = len(images)
    nImgCols = len(images[0])
    fullWidth = nImgCols * img_shape[1]
    imgArray = np.zeros([(img_shape[0]+padding) * int(0.5 + float(nImgRows)/float(featuresPerRow)) - padding,
                         (fullWidth+padding) * featuresPerRow - padding], dtype=np.uint8)
    PILshape = [imgArray.shape[1],imgArray.shape[0]]

    iRow = 0
    iFeature = 0
    for row in images:
        for iImgCol in range(0,nImgCols):
            startY = iRow*(img_shape[0]+padding)
            startX = iFeature*(fullWidth+padding) + iImgCol*img_shape[1]
            imgArray[startY:startY+img_shape[0], startX:startX+img_shape[1]] = \
                np.array(np.reshape(row[iImgCol].clip(min=-1.0, max=1.0), img_shape) * 127.5 + 127.5, dtype=np.uint8)
        iFeature += 1
        if iFeature >= featuresPerRow:
            iRow += 1
            iFeature = 0

        Image.frombuffer('L', PILshape, imgArray.tostring('C'), 'raw', 'L', 0, 1).save(path)
