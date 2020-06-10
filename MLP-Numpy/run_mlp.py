from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import numpy as np
import os

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
# model = Network()
# model.add(Linear('fc1', 784, 10, 0.001))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

# config = {
#     'learning_rate': 0.000001,
#     'weight_decay': 0.005,
#     'momentum': 0.9,
#     'batch_size': 100,
#     'max_epoch': 100,
#     'disp_freq': 50,
#     'test_epoch': 5
# }


# no hidden layers, fc + relu
def zero_layer_relu():
    model = Network()
    model.add(Linear('fc1', 784, 10, 0.01))
    model.add(Relu('rl1'))
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 50,
        'max_epoch': 20,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config


# one hidden layer, fc[784->256] + sigmoid + fc[256->10] + sigmoid
def one_layer_sigmoid():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Sigmoid('sg2'))
    config = {
        'learning_rate': 0.05,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config


# one hidden layer, fc[784->256] + relu + fc[256->10] + relu
def one_layer_relu():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Relu('rl2'))
    config = {
        'learning_rate': 0.05,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 200,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config


# two hidden layer, fc[784->500] + sigmoid + fc[500->256] + sigmoid + fc[256->10] + sigmoid
def two_layer_sigmoid():
    model = Network()
    model.add(Linear('fc1', 784, 500, 0.01))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 500, 256, 0.01))
    model.add(Sigmoid('sg2'))
    model.add(Linear('fc3', 256, 10, 0.01))
    model.add(Sigmoid('sg3'))
    config = {
        'learning_rate': 0.01,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config


# two hidden layer, fc[784->500] + relu + fc[500->256] + relu + fc[256->10] + relu
def two_layer_relu():
    model = Network()
    model.add(Linear('fc1', 784, 500, 0.01))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 500, 256, 0.01))
    model.add(Relu('rl2'))
    model.add(Linear('fc3', 256, 10, 0.01))
    model.add(Relu('rl3'))
    config = {
        'learning_rate': 0.05,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 200,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config


def run(net_func, save_loss_path, save_acc_path, result_dir="result/"):
    model, config = net_func()
    loss_, acc_ = [], []

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        a, b = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        loss_ += a
        acc_ += b
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'])
    test_net(model, loss, test_data, test_label, config['batch_size'])

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    np.save(result_dir + save_loss_path, loss_)
    np.save(result_dir + save_acc_path, acc_)


result_dir = "result/"

#run(net_func=zero_layer_relu, save_loss_path="loss0r", save_acc_path="acc0r")

#run(net_func=one_layer_sigmoid, save_loss_path="loss1s", save_acc_path="acc1s")

#run(net_func=one_layer_relu, save_loss_path="loss1r", save_acc_path="acc1r")

#run(net_func=two_layer_sigmoid, save_loss_path="loss2s", save_acc_path="acc2s")

run(net_func=two_layer_relu, save_loss_path="loss2r", save_acc_path="acc2r")
