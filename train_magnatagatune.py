#!/usr/bin/env python
"""

Script to train a 2D convolutional neural network on the Magnatagatune
dataset, with log-amplitude mel spectral features as input.

Works with Keras version: 1.2.2
"""
import argparse
import datetime
import math
import os
import keras
from keras.layers.convolutional import Convolution1D, Convolution2D, Conv2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils, generic_utils
import numpy as np
from scipy import ndimage
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser('Train a 2D conv model on Magnatagatune Mel Features.')
parser = argparse.ArgumentParser(description='Create a train/valid/test split for Magnatagatune.')
parser.add_argument('--dataset_dir',
                    help='The directory with the output of create_train_valid_test_split.py.')
parser.add_argument('--audio_dir',
                    help='The directory containing Magnatagatune audio subdirectories.')
parser.add_argument('--model_dir',
                    help='The directory to save the model tox.')

args = parser.parse_args()

# Load data set.
dataset_dir = args.dataset_dir
labelmap_fname = os.path.join(dataset_dir, 'labelmap.txt')
train_fname = os.path.join(dataset_dir, 'train.tsv')
valid_fname = os.path.join(dataset_dir, 'valid.tsv')
test_fname = os.path.join(dataset_dir, 'test.tsv')
with open(labelmap_fname) as fi:
    lines = [line.strip() for line in open(labelmap_fname)]
    label_map = {tag: i for i, tag in enumerate(lines)}


def pairs2xy(pairs):
    """
    Takes pairs of (filename, tag) and output training pairs of
    (filename, one-hot).
    """
    # X is the train file paths
    # y is the label (which we convert from tag to numerical to one-hot)
    Xl = [ex[0] for ex in pairs]
    yl = [ex[1] for ex in pairs]
    yl = np_utils.to_categorical([label_map[tag] for tag in yl])
    return Xl, yl


def pairs2xy_many_label(pairs):
    """
    Rather than outputting X,y pairs with non-unique X values
    and one-hot y's, we output a shorter length number of pairs
    with unique X values and "many-hot" y's. This is used to
    evaluate AUC.
    """
    # X is the train file paths
    # y is the label (which we convert from tag to numerical to one-hot)
    Xl, yl =  pairs2xy(pairs)
    fname_to_many_label_one_hot = {}
    for exx, exy in zip(Xl, yl):
        if exx in fname_to_many_label_one_hot:
            fname_to_many_label_one_hot[exx] += exy
        else:
            fname_to_many_label_one_hot[exx] = exy
    # The lines below assume lists are returned (Python 2)
    Xl = fname_to_many_label_one_hot.keys()
    yl = fname_to_many_label_one_hot.values()
    return Xl, yl
train_examples = [line.strip().split('\t') for line in open(train_fname)]
valid_examples = [line.strip().split('\t') for line in open(valid_fname)]
test_examples = [line.strip().split('\t') for line in open(test_fname)]
X_train, y_train = pairs2xy(train_examples)
X_valid, y_valid = pairs2xy(valid_examples)
X_test, y_test = pairs2xy(test_examples)
X_valid_many_label, y_valid_many_label = pairs2xy_many_label(valid_examples)

print(train_examples[0])
print(X_train[0])
print(y_train[0])
print(len(y_train[0]))

dirname = args.audio_dir


def audio_fname_to_mel_fname(audio_fname):
    return os.path.join(dirname,
                        audio_fname.replace('.mp3',
                                            '.mp3.mel.npy')
                       )


def load_mel_file(fname):
    """
    If we just load np.load the mel file we get a shape of
    (1, 1, 96, 1366) == batch, kernel, freq, time
    
    We load data in the get_xy() function which instantiates
    data by calling np.array(list) rather than np.concatenate,
    so we don't want the initial batch dimension.
    
    We also want the kernel dim as the last dimension, since we
    use data_format="channels_last" in convolution2d().
    """
    return np.load(
        audio_fname_to_mel_fname(fname)
    ).reshape(
        (96, 1366, 1)
    )


def get_xy(X_in, y_in, load_function, num_per_yield=64*10):
    num_yields = int(math.ceil(len(X_in) / float(num_per_yield)))
    for i in range(num_yields):
        begin_index = i * num_per_yield
        end_index = min((i + 1) * num_per_yield,
                        len(X_in))
        x_out = np.array(
            [load_function(fname) 
             for fname in X_in[begin_index : end_index]]
        )
        yield x_out, y_in[begin_index : end_index]


def get_xy_sample(X_in, y_in, load_function, num_per_yield=64*10):
    """
    This version never runs out. It randomly samples num_per_yield
    examples forever.
    """
    while True:
        sample_indices = np.random.permutation(len(y_in))[:num_per_yield]
        x_out = np.array(
            [load_function(X_in[i])
             for i in sample_indices]
        )
        y_out = np.array([
            y_in[i] for i in sample_indices]
        )
        yield x_out, y_out

print('number input examples: {}'.format(len(y_train)))
ie_mel = load_mel_file(X_train[0])
print('Shape of subsampled mel input example: {}'.format(ie_mel.shape))
print('number features per subsampled mel input example: {}'.format(len(ie_mel.flatten())))


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
def print_stats(predicted, real, proba):
    """
        predicted: list of indices of predicted classes
        real: list of 1-hot ground truth vector labels
        proba: list of probability vectors
    """
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    auc = roc_auc_score(real, proba)
    print('AUC: {}'.format(auc))
    return auc


def get_mel_network():
    """
    Get a network that trains on mel features.
    This is based on the FCN-4 architecture of Choi et al.
    ("Automatic tagging using deep convolutional neural networks")
    https://arxiv.org/abs/1606.00298
    
    Rectified Linear Unit (ReLU) is
    used as an activation function in every convolutional layer
    except the output layer, which uses Sigmoid to squeeze the
    output within [0, 1]. Batch Normalisation is added after every
    convolution and before activation [11]. Dropout of 0.5
    is added after every max-pooling layer [24]. This accelerates
    the convergence while dropout prevents the network
    from overfitting.

    Keunwoo admits that he used way too many feature maps in the paper.
    He says 32 each will get the same performance.
    Also, Dropout of 0.5 is a bit aggressive.
    https://keunwoochoi.wordpress.com/2017/01/12/a-self-critic-on-my-ismir-paper-automatic-tagging-using-deep-convolutional-neural-networks/
    
    On input format: 
    
    
    data_format: A string, one of channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape
    (batch, width, height, channels) while channels_first corresponds to inputs with shape
    (batch, channels, width, height).
    
    It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json.
    If you never set it, then it will be "channels_last".
    
    https://keras.io/layers/convolutional/#convolution2d
    
    Keunwoo uses
        freq_axis == 1
        time_axis == 2
        channel_axis == 3
        
        This is from https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_cnn.py,
        which is slightly different from the paper (eg. uses ELU instead of ReLu)
    
    """
    global ie_mel

    modmel = Sequential()
    # Shape of mel input example: (96, 1366, 1)
    #input_shape = ie_mel.shape
    input_shape = ie_mel.shape
    print(input_shape)
    
    #96 == #mel-bins and 1366 == #time frame
    batch_axis = 0
    freq_axis = 1
    time_axis = 2
    channel_axis = 3  # This is channel_last / tf / TensorFlow data_format
    
    def add_conv(m, num_channels, pool_size=None, input_shape=None):
    
        if input_shape:
            m.add(Convolution2D(#nb_filter=128,
                                #kernel_size=(3,3),
                                num_channels,
                                3,
                                3,
                                border_mode='valid',
                                input_shape=input_shape,
                                #data_format='channels_last'
                               )
                  )
        else:
            m.add(Convolution2D(num_channels,
                                3,
                                3,
                                border_mode='valid',
                               )
                  )

        m.add(BatchNormalization(axis=channel_axis, mode=0))
        m.add(Activation('relu'))
        
        if pool_size:
            m.add(
                MaxPooling2D(
                    pool_size=pool_size,
                    #padding='valid'
                )
            )
        m.add(Dropout(0.25))

    # Pool here because my 8 GB memory 1070 GTX can't handle features this large
    # with batch size of 32.
    modmel.add(
        MaxPooling2D(
            pool_size=(2,2),
            input_shape=input_shape
            #padding='valid'
        )
    )
    add_conv(modmel, 32, (1,2))
    add_conv(modmel, 32, (1,2))
    add_conv(modmel, 32, (2,4))
    # Pool here to reduce dimensionality of conv -> dense matrix.
    add_conv(modmel, 32, (2,4))
    
    modmel.add(Flatten())
    modmel.add(Dense(output_dim=len(y_train[0]), init="uniform"))
    #modmel.add(Activation("softmax"))
    # Sigmoid is used in Keunwoo's paper.
    modmel.add(Activation("sigmoid"))
    
    return modmel


def train_model_by_time(m, time_in_seconds, load_fn, batch_size=32):
    global X_train
    global y_train
    
    num_yields_train = 100
    num_per_yield_train = len(y_train) / num_yields_train
    print('Number of training points: {}'.format(len(y_train)))
    print('Number of training points loaded into memory at a time: {}'.format(num_per_yield_train))
    
    sn_train_gener = get_xy_sample(X_train, y_train, load_fn, num_per_yield=num_per_yield_train)
    
    i = 0
    time_out = False
    st = datetime.datetime.now()
    run_time = 0
    while not time_out:
        i += 1
        try:
            X_train_mem, y_train_mem = next(sn_train_gener)
        except Exception as e:
            print(e)
            print('Done with Training Data (this should not happen)')
            break
        # Only print progress bar and metrics on last mini-epoch of the full epoch.
        # Mini-epochs are used because data cannot all fit into memory.
        if i % num_yields_train == 0:
            m.fit(X_train_mem, y_train_mem, nb_epoch=1, batch_size=batch_size)
        else:
            m.fit(X_train_mem, y_train_mem, nb_epoch=1, batch_size=batch_size, verbose=0)
            
        et = datetime.datetime.now()
        run_time = (et-st).total_seconds()
        time_out = (run_time > time_in_seconds)
    # Train one last time, with verbose off to ensure you print stats. 
    X_train_mem, y_train_mem = next(sn_train_gener)
    m.fit(X_train_mem, y_train_mem, nb_epoch=1, batch_size=batch_size)
    #print('Ran for {} seconds.'.format(run_time))
    print('Ran for {} iterations.'.format(i))


def test_model(m, load_fn, x_data, y_data, batch_size=32):
    num_yields_test = 10
    num_per_yield_test=int(math.ceil(len(y_data)/num_yields_test))
    print('Number of test points: {}'.format(len(y_data)))
    print('Number of test batches: {}'.format(num_yields_test))
    print('Number of test points loaded into memory at a time: {}'.format(num_per_yield_test))

    sn_test_gener = get_xy(x_data, y_data, load_fn, num_per_yield=num_per_yield_test)
    proba_test = None
    classes_test = None
    y_test_concat = None

    for _ in range(num_yields_test):
        try:
            X_test_mem, y_test_mem = next(sn_test_gener)
        except Exception as e:
            print(e)
            print('Done with test data.')
            break

        proba_test_part = m.predict_proba(X_test_mem, batch_size=batch_size)
        classes_test_part = np.round(proba_test_part)

        if proba_test is None:
            proba_test = proba_test_part
        else:
            proba_test = np.concatenate((proba_test, proba_test_part), axis=0)
        if classes_test  is None:
            classes_test = classes_test_part
        else:
            classes_test = np.concatenate((classes_test, classes_test_part))
        if y_test_concat is None:
            y_test_concat = y_test_mem
        else:
            y_test_concat = np.concatenate((y_test_concat, y_test_mem))

    print('')
    return print_stats(classes_test, y_test_concat, proba_test)

model_dir = args.model_dir
model_fname = os.path.join(model_dir, 'keunwoo_magna.h5')

if os.path.exists(model_fname):
    print('Model already exists, loading from memory.')
    m = load_model(model_fname)
else:
    print('Model not found on disk, starting from scratch.')
    m = get_mel_network()
    m.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=["accuracy"])
print(m.summary())

print('Training!')
patience = 5
num_runs_no_improvement = 0
best_valid_auc = 0
while True:
    train_model_by_time(m,
                        time_in_seconds=60*2,
                        load_fn=load_mel_file,
                        batch_size=32)
    print('Done training (for now). Now evaluating on validation set.')
    st_valid = datetime.datetime.now()
    valid_auc = test_model(m,
                           load_mel_file,
                           X_valid_many_label,
                           y_valid_many_label,
                           batch_size=32)
    et_valid = datetime.datetime.now()
    print('Validation took {}'.format(str(et_valid - st_valid)))
    # Evaluate early stopping criteria.
    if valid_auc <= best_valid_auc:
        num_runs_no_improvement += 1
    else:
        num_runs_no_improvement = 0
    best_valid_auc = max(best_valid_auc, valid_auc)
    print('Saving model...')
    m.save(model_fname)  # creates a HDF5 file '{model_name}.h5'
    print('Model saved!')
    if num_runs_no_improvement >= patience:
        print('Validation statistics have not improved for {} steps'.format(
              num_runs_no_improvement)
        )
        print('Early stopping criteria met. Stopping training.')
        break

