import cPickle
import gzip
import os
import sys
import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

def load_data(dataset='../../../data/mnist.pkl.gz'):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        data_x -= np.mean(data_x, axis =1, keepdims = True)
        data_x /=  (np.std(data_x, axis = 1, keepdims = True) + 1e-5)
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def rectify(X):
    return T.maximum(X, 0.)

def sgd(cost, params, learning_rate = 0.01):
    grads = T.grad(cost, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

# Lasagne's version of classic momentum
def apply_momentum(updates, params=None, momentum=0.9):
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),name = "v", broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates

def lasagne_momentum(cost, params, learning_rate=0.01, momentum=0.9):
    updates = sgd(cost, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)

# My implement of classic momentum
def momentum(cost, params, learning_rate =0.01, momentum=0.9):
    grads =  T.grad(cost, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),broadcastable=param.broadcastable)
        updates[velocity] = momentum*velocity - learning_rate * grad
        updates[param] = param + updates[velocity]
    #(variable, update expression) pairs
    return updates

# Lasagne's version of NAG (Nesterov Momentum)
def apply_nag(updates, params=None, momentum=0.9):
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), name = "v", broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates

def lasagne_nag(cost, params, learning_rate=0.01, momentum=0.9):
    updates = sgd(cost, params, learning_rate)
    return apply_nag(updates, momentum=momentum)

# My implement of NAG (Nesterov Momentum)
def nag(cost, params, learning_rate =0.01, momentum=0.9):
    grads =  T.grad(cost, params)
    updates=OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), name="v", broadcastable=param.broadcastable)
        updates[param] = param - learning_rate * grad
        updates[velocity] = momentum * velocity + updates[param] - param
        updates[param] = momentum * updates[velocity] + updates[param]
    return updates


class ProcessRecorder(object):
    def __init__(self):
        self.epoch = []
        self.train_cost = []
        self.train_accuracy = []
        self.vali_cost = []
        self.vali_accuracy = []
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d
    def save_recorder(self, train_recorder, path="../model/lr_train_recorder.pkl"):
        best_recorder_string = cPickle.dumps(train_recorder)
        with open(path, "wb") as handle:
            handle.write(best_recorder_string)
    def load_recorder(self, path="../model/lr_train_recorder.pkl"):
        with open(path, "rb") as handle:
            train_recorder_string = handle.read()
        train_recorder = cPickle.loads(train_recorder_string)
        return train_recorder

class BestRecorder(object):
    def __init__(self):
        self.best_W_h = None
        self.best_b_h = None
        self.best_W_o = None
        self.best_b_o = None
        self.best_vali_cost = np.inf
        self.best_vali_epoch = 0
        self.best_vali_acc = 0.
        self.best_weights_path = "../model/lr_model.pkl"
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d
    def save_recorder(self, best_recorder, path="../model/lr_best_recorder.pkl"):
        best_recorder_string = cPickle.dumps(best_recorder)
        with open(path, "wb") as handle:
            handle.write(best_recorder_string)
    def load_recorder(self, path="../model/lr_best_recorder.pkl"):
        with open(path, "rb") as handle:
            best_recorder_string = handle.read()
        best_recorder = cPickle.loads(best_recorder_string)
        return best_recorder

def early_stopping(parameters, best_recorder, current_acc, current_vali_cost, current_epoch, patience = 500):
    if current_vali_cost < best_recorder.best_vali_cost:
        best_recorder.best_vali_cost = current_vali_cost
        best_recorder.best_vali_epoch = current_epoch
        best_recorder.best_vali_acc = current_acc
        W_h, b_h, W_o, b_o = parameters
        best_recorder.best_W_h = W_h.get_value()
        best_recorder.best_b_h = b_h.get_value()
        best_recorder.best_W_o = W_o.get_value()
        best_recorder.best_b_o = b_o.get_value()
        return False
    elif best_recorder.best_vali_epoch + patience < current_epoch:
        return True

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(x, W_h, b_h, W_o, b_o, training = True):
    # Activation function
    h = T.nnet.sigmoid(T.dot(x, W_h) + b_h.dimshuffle('x',0))
    if training:
        h = dropout(h, 0.5)
    else:
        h = 0.5*h
    # Softmax
    p_y_given_x = T.nnet.softmax(T.dot(h,W_o)+b_o.dimshuffle('x',0))
    return p_y_given_x


def train(LOAD_MODEL = False):
    # Parameters
    n_in = 28*28
    n_h = 600
    n_out = 10
    batch_size = 100

    # Load Data
    datasets = load_data()
    trX, trY = datasets[0]
    vaX, vaY = datasets[1]
    teX, teY = datasets[2]

    n_train_batches = trX.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = vaX.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = teX.get_value(borrow=True).shape[0] / batch_size

    # Weights & Recorder initialization
    if LOAD_MODEL == True:
        MODEL_PARAMS_PATH = "model/lr_model.pkl"
        BEST_REC_PATH = "model/lr_best_recorder.pkl"
        TRAIN_REC_PATH = "model/lr_train_recorder.pkl"
        train_recorder = ProcessRecorder().load_recorder()
        best_recorder = BestRecorder().load_recorder()
        W = best_recorder.best_W
        b = best_recorder.best_b
        W = theano.shared(W.astype(theano.config.floatX),  name="W")
        b = theano.shared(b.astype(theano.config.floatX), name="b")

    else:
        #W = theano.shared( value=np.zeros((n_in, n_out)).astype(theano.config.floatX), name='W', borrow=True)
        #b = theano.shared( value=np.zeros((n_out,)).astype(theano.config.floatX), name='b', borrow=True)
        W_h = theano.shared( value= np.random.randn(n_in, n_h).astype(theano.config.floatX), name='W_h', borrow=True)
        b_h = theano.shared( value= np.random.randn(n_h,).astype(theano.config.floatX), name='b_h', borrow=True)
        W_o = theano.shared( value= np.random.randn(n_h, n_out).astype(theano.config.floatX), name='W_o', borrow=True)
        b_o = theano.shared( value= np.random.randn(n_out,).astype(theano.config.floatX), name='b_o', borrow=True)
        #W = theano.shared( value=np.random.randn(n_in, n_out).astype(theano.config.floatX), name='W', borrow=True)
        #b = theano.shared( value=np.random.randn(n_out,).astype(theano.config.floatX), name='b', borrow=True)
        train_recorder = ProcessRecorder()
        best_recorder = BestRecorder()


    # Symbolic Expression definition
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    p_y_given_x = model(x=x, W_h=W_h, b_h=b_h, W_o=W_o, b_o=b_o, training=True)
    # Cost
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    #cost = T.mean(T.nnet.categorical_crossentropy( p_y_given_x, y))
    # Error
    p_y_given_x_pred = model(x=x, W_h=W_h, b_h=b_h, W_o=W_o, b_o=b_o, training=False)
    y_pred = T.argmax(p_y_given_x_pred, axis=1)
    acc = T.mean(T.eq(y_pred, y))
    cost_pred = -T.mean(T.log(p_y_given_x_pred)[T.arange(y.shape[0]), y])
    error = T.mean(T.neq(y_pred, y))
    params = [W_h, b_h, W_o, b_o]
    #updates = sgd(cost, params)
    updates = momentum(cost, params)
    #updates= lasagne_momentum(cost, params)
    #updates = lasagne_nag(cost, params)
    #updates = nag(cost, params)

    # Theano Function
    train_model = theano.function(inputs=[index], outputs = cost, updates = updates,
                                  givens={x: trX[index * batch_size: (index + 1) * batch_size],
                                          y: trY[index * batch_size: (index + 1) * batch_size]})

    predict_model = theano.function(inputs=[index],outputs=acc, updates = updates,
                                    givens={x: trX[index * batch_size: (index + 1) * batch_size],
                                            y: trY[index * batch_size: (index + 1) * batch_size]})

    vali_acc_model = theano.function(inputs=[], outputs=acc, givens={x: vaX, y: vaY})
    vali_cost_model = theano.function(inputs=[], outputs = cost_pred, givens={x: vaX, y: vaY})

    test_acc_model = theano.function(inputs=[], outputs=acc, givens={x: teX, y: teY})
    test_cost_model = theano.function(inputs=[], outputs = cost_pred, givens={x: teX, y: teY})

    # Start Training
    epoch = 0
    while (epoch < 500):
        epoch = epoch + 1
        minibatch_avg_cost = 0
        minibatch_avg_acc = 0

        # Mini-Batch
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost += train_model(minibatch_index)
            minibatch_avg_acc += predict_model(minibatch_index)

        trCost = minibatch_avg_cost / n_train_batches
        trAcc = minibatch_avg_acc / n_train_batches
        vaCost = vali_cost_model()
        vaAcc = vali_acc_model()
        teCost = test_cost_model()
        teAcc = test_acc_model()
        print "epoch %i" % epoch
        print "Training Cost: ", trCost
        print "Training Accuracy: ", trAcc
        print "Validation Cost: ", vaCost
        print "Validation Accuracy: ", vaAcc
        print "Testing Cost: ", teCost
        print "Testing Accuracy: ", teAcc
        print "\n"

        train_recorder.epoch.append(epoch)
        train_recorder.train_accuracy.append(trAcc)
        train_recorder.train_cost.append(trCost)
        train_recorder.vali_accuracy.append(vaAcc)
        train_recorder.vali_cost.append(vaCost)

        # Early Stopping
        isEarlyStopping = early_stopping(parameters=params, best_recorder=best_recorder, current_acc=vaAcc,
                                         current_vali_cost= vaCost, current_epoch= epoch, patience = 500)
        if isEarlyStopping:
            print "Early Stopped!!!"
            break

    # Save recorder
    best_recorder.save_recorder(best_recorder)
    train_recorder.save_recorder(train_recorder)

if __name__ == '__main__':
    train()
