# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

FLAGS = None


def model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32, [])
    y_ = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, [])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
        conv1 = slim.conv2d(x_image, 32, [5, 5], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.conv2d(pool1, 64, [5, 5], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
        flatten = slim.flatten(pool2)
        fc = slim.fully_connected(flatten, 1024, scope='fc1')
        drop = slim.dropout(fc, keep_prob=keep_prob)
        logits = slim.fully_connected(drop, 10, activation_fn=None, scope='logits')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)

    return {'x': x,
            'y_': y_,
            'keep_prob': keep_prob,
            'is_training': is_training,
            'train_step': train_op,
            'accuracy': accuracy,
            'cross_entropy': cross_entropy}


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    net = model()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # Train
    for _ in range(201):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # slim.learning.train(train_op, FLAGS.train_log_dir)
        sess.run([net['train_step']], feed_dict={net['x']: batch_xs,
                                                 net['y_']: batch_ys,
                                                 net['keep_prob']: 0.5,
                                                 net['is_training']: True})
        if _ % 50 == 0:
            entropy, acc = sess.run([net['cross_entropy'], net['accuracy']],
                                    feed_dict={net['x']: batch_xs,
                                               net['y_']: batch_ys,
                                               net['keep_prob']: 1.0,
                                               net['is_training']: True})
            print('step {}: entropy {}: accuracy {}'.format(_, entropy, acc))
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'))
    print('Finish training')

    # validation
    acc = 0.0
    for i in range(50):
        batch_xs, batch_ys = mnist.validation.next_batch(100)
        acc_ = sess.run(net['accuracy'], feed_dict={net['x']: batch_xs,
                                                    net['y_']: batch_ys,
                                                    net['keep_prob']: 1.0,
                                                    net['is_training']: False})
        acc += acc_
    print('Overall validation accuracy {}'.format(acc / 50))


def test():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # Test trained model
    net = model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print("restore from the checkpoint {0}".format(ckpt))

    acc = 0.0
    for i in range(50):
        batch_xs, batch_ys = mnist.test.next_batch(100)
        acc_ = sess.run(net['accuracy'], feed_dict={net['x']: batch_xs,
                                                    net['y_']: batch_ys,
                                                    net['keep_prob']: 1.0,
                                                    net['is_training']: False})
        acc += acc_
    print('Overall test accuracy {}'.format(acc / 50))


def main(_):
    if FLAGS.phase == 'train':
        train()
    else:
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data',
                        help='Directory for storing input data')
    parser.add_argument('--phase', type=str, default='test',
                        help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--train_log_dir', type=str, default='log',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory for checkpoint file')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
