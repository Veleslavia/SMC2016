import numpy as np
import theano
import theano.tensor as T
import lasagne

import pickle

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils import load, visualization

instruments_mapping = {"accordion": "n02672831",
                       "banjo": "n02787622",
                       "cello": "n02992211",
                       "trumpet": "n03110669",
                       "drum": "n03249569",
                       "flute": "n03372029",
                       "guitar": "n03467517",
                       "oboe": "n03838899",
                       "piano": "n03928116",
                       "saxophone": "n04141076",
                       "trombone": "n04487394",
                       "violin": "n04536866"}

def build_model():
    net = dict()
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == N:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 64


def train_batch(X_tr, y_tr, train_fn):
    ix = range(len(y_tr))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return train_fn(X_tr[ix], y_tr[ix])


def val_batch(X_val, y_val, val_fn):
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])


def predict_batch(X_data, pred_fn):
    y_pred = []
    for chunk in batches(X_data, BATCH_SIZE):
        y_pred.extend(pred_fn(chunk).argmax(-1))
    y_pred = np.reshape(y_pred, -1)
    return y_pred


def load_model():
    d = pickle.load(open('./cnnet/vgg16.pkl'))
    #d = pickle.load(open('vgg16_finetuned_2_0.748888888889.pkl'))

    net = build_model()
    lasagne.layers.set_all_param_values(net['prob'], d['param values'])

    # The network expects input in a particular format and size.
    # We define a preprocessing function to load a file and apply the necessary transformations
    IMAGE_MEAN = d['mean value'][:, np.newaxis, np.newaxis]

    return IMAGE_MEAN, net


def redesign_model(net):
    # We'll connect our output classifier to the last fully connected layer of the network
    output_layer = DenseLayer(net['fc7'], num_units=12, nonlinearity=softmax)   # 12 classes
    return output_layer


def new_model_params(cnn_model):
    # Define loss function and metrics, and get an updates dictionary
    X_sym = T.tensor4()
    y_sym = T.ivector()

    prediction = lasagne.layers.get_output(cnn_model, X_sym)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
    loss = loss.mean()

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                          dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(cnn_model, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)


    # Compile functions for training, validation and prediction
    train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
    val_fn = theano.function([X_sym, y_sym], [loss, acc])
    pred_fn = theano.function([X_sym], prediction)
    return train_fn, val_fn, pred_fn


def train_one_net(cnn_net, train_fn, val_fn, pred_fn, X_tr, y_tr, X_val, y_val, X_te, y_te, fold_number=0):
    acc_tot = 0.
    for epoch in range(5):
        for batch in range(25):
            loss = train_batch(X_tr, y_tr, train_fn)

        ix = range(len(y_val))
        np.random.shuffle(ix)

        loss_tot = 0.
        acc_tot = 0.
        for chunk in batches(ix, BATCH_SIZE):
            loss, acc = val_fn(X_val[chunk], y_val[chunk])
            loss_tot += loss * len(chunk)
            acc_tot += acc * len(chunk)

        loss_tot /= len(ix)
        acc_tot /= len(ix)
        print(epoch, loss_tot, acc_tot * 100)
    pickle.dump(cnn_net, open("finetuned_net_fold{number}.plk".format(number=fold_number), "w"))

    p_y = predict_batch(X_val, pred_fn)
    pickle.dump(p_y, open("val_predict_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_val, open("val_eval_fold{number}.plk".format(number=fold_number), "w"))
    print p_y
    print y_val
    try:
        print confusion_matrix(y_val, p_y)
        visualization.plot_confusion_matrix(confusion_matrix(y_val, p_y),
                                            title="Validation CM fold{number}".format(number=fold_number),
                                            labels=instruments_mapping.keys())
        print classification_report(y_val, p_y)
    except:
        pass

    p_y = predict_batch(X_te, pred_fn)
    pickle.dump(p_y, open("te_predict_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_te, open("te_eval_fold{number}.plk".format(number=fold_number), "w"))
    print p_y
    print y_te
    try:
        print confusion_matrix(y_te, p_y)
        visualization.plot_confusion_matrix(confusion_matrix(y_te, p_y),
                                            title="Test CM fold{number}".format(number=fold_number),
                                            labels=instruments_mapping.keys())
        print classification_report(y_te, p_y)
    except:
        pass


def train_model():

    IMAGE_MEAN, base_model = load_model()

    X, y = load.load_data(IMAGE_MEAN)
    skf = StratifiedKFold(y, n_folds=5, shuffle=True)
    for fold_number, train_index, test_index in enumerate(skf):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_te, y_te = X[test_index], y[test_index]
        train_index, val_index = train_test_split(train_index)
        X_tr, y_tr = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        index = {"train": train_index, "val": val_index, "test": val_index}
        pickle.dump(index, open("index_fold{number}.plk".format(number=fold_number), "w"))

        cnn_model = redesign_model(base_model)
        train_fn, val_fn, pred_fn = new_model_params(cnn_model)
        train_one_net(cnn_model, train_fn, val_fn, pred_fn, X_tr, y_tr, X_val, y_val, X_te, y_te, fold_number)

    # X_tr, y_tr, X_val, y_val, X_te, y_te = load.load_dataset_transfer(IMAGE_MEAN)