from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net, getvis
from load_data import load_mnist_4d
import numpy as np
import cv2


# import matplotlib.pyplot as plt
def vis_square(data, id):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # print(data)
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    cv2.imwrite('try%d.png'%id, data*255)
    # print(data.shape)
    # print(data)
    # plt.savefig('try%d.pdf'%id)
    # plt.imshow(data)
    # plt.axis('off')


train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
# Batch = N x 28 x 28

# Origin CNN
model = Network()
model.add(Conv2D('conv1', in_channel=1, out_channel=4, kernel_size=3, pad=1, init_std=0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', kernel_size=2, pad=0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', in_channel=4, out_channel=4, kernel_size=3, pad=1, init_std=0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', kernel_size=2, pad=0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', in_num=196, out_num=10, init_std=0.1))

'''
# LeNet
model = Network()
model.add(Conv2D('conv1', in_channel=1, out_channel=6, kernel_size=5, pad=1, init_std=0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', kernel_size=2, pad=0))  # output shape: N x 14 x 14 x 6
model.add(Conv2D('conv2', in_channel=6, out_channel=16, kernel_size=5, pad=1, init_std=0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', kernel_size=2, pad=0))  # output shape: N x 5 x 5 x 16 = N x 400
#model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', in_num=400, out_num=120, init_std=0.1))
model.add(Relu('relu3'))
model.add(Linear('fc4', in_num=120, out_num=10, init_std=0.1))
'''

# loss = EuclideanLoss(name='loss')
loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

# np.random.seed(1626)
config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 10,
    'test_epoch': 5,
    'lr_epochs': 4,
}
loss_ = []
acc_ = []
test_loss = np.zeros(2) + 100
for epoch in range(config['max_epoch']):
    vis = getvis(model, test_data, test_label)
    for i in range(4):
        vis_square(vis[i], i)
    vis_square(model.layer_list[0].W.reshape(-1, model.layer_list[0].W.shape[2], model.layer_list[0].W.shape[3]), -1)
    LOG_INFO('Training @ %d epoch...' % (epoch))
    a, b = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], 600)
    loss_.append(a)
    acc_.append(b)
    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_ = test_net(model, loss, test_data, test_label, 100)
        if epoch % config['lr_epochs'] == 0 and epoch > 0:
        # if test_loss.min() < test_ and abs(test_loss.max() - test_loss.min()) / test_loss.min() < 0.01 or epoch % 7 == 0:
            config['learning_rate'] /= 2
            config['weight_decay'] = 0
            print('lr: ', config['learning_rate'])
        test_loss[1:] = test_loss[:-1].copy()
        test_loss[0] = test_
vis = getvis(model, test_data, test_label)
for i in range(4):
    vis_square(vis[i], i)
np.save('loss', loss_)
np.save('acc', acc_)
