# -*- coding: utf-8 -*-

try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import os
from tensorflow.python.training import moving_averages
FLAGS = tf.app.flags.FLAGS  # for command args


class Model(object):
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        # take for place and it will run in sess()
        with tf.name_scope('input'):
            self.x_ = tf.placeholder(tf.float32, [None, 28*28], name='x_in')
            self.y_ = tf.placeholder(tf.int32, [None], name='y_in')
        # self.hidden_size = FLAGS.hidden_size
        self.hidden_size = 256
        # initial weight in random(uniform = True, 均匀)(normal :正态)
        # self.init = tf.contrib.layers.xavier_initializer()
        self.init = tf.truncated_normal_initializer(stddev=0.1)
        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc, self.train_summary_op = self.forward(is_train=True, reuse=False)
        self.loss_val, self.pred_val, self.acc_val, self.valid_summary_op = self.forward(is_train=False, reuse=True)

        # 用于对变量进行更新，包括变量的value和shape
        # 变量assign之后才能进行使用
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)  # for traing step
        self.params = tf.trainable_variables()  # for all params

        # TODO:  maybe you need to update the parameter of batch_normalization ?
        # 使用上下文管理处理依赖关系: 在所有变量都更新之后，使用train_op操作
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step, # update global_step for updating lr
                                                                                var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
        with tf.variable_scope("model", reuse=reuse):  # 创建变量作用域
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer (fully-connect)
            fc1 = tf.layers.dense(inputs=self.x_, units=self.hidden_size, kernel_initializer=self.init)
            # Your BN Layer: use batch_normalization_layer function
            bn1 = batch_normalization_layer(fc1, is_train)
            # Your Relu Layer
            relu1 = tf.nn.relu(bn1)
            # Your Dropout Layer: use dropout_layer function
            drop1 = dropout_layer(relu1, FLAGS.drop_rate, is_train)
            # Your Linear Layer
            logits = tf.layers.dense(inputs=drop1, units=10, kernel_initializer=self.init)
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        # labels neddn't to be one-hot, it will transfrom automatically
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, axis=1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        # summary
        train_loss_sum = tf.summary.scalar('loss', loss)
        train_acc_sum = tf.summary.scalar('acc', acc)
        summary_op = tf.summary.merge([train_loss_sum, train_acc_sum])
        return loss, pred, acc, summary_op


def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    # axis 是进行规范化的维度
    return tf.layers.batch_normalization(incoming, axis=1, training=is_train)


def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    return tf.layers.dropout(incoming, drop_rate, training=is_train)
    # return tf.nn.dropout(incoming, keep_prob=1-drop_rate) if is_train else incoming
