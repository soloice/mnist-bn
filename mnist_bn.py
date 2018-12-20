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
from tensorflow.python import debug as tf_debug

FLAGS = None


def model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32, [])
    y_ = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, [])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training, 'momentum': 0.95}):
        conv1 = slim.conv2d(x_image, 16, [5, 5], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.conv2d(pool1, 32, [5, 5], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
        flatten = slim.flatten(pool2)
        fc = slim.fully_connected(flatten, 1024, scope='fc1')
        drop = slim.dropout(fc, keep_prob=keep_prob)
        logits = slim.fully_connected(drop, 10, activation_fn=None, scope='logits')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        print("BN parameters: ", update_ops)
        updates = tf.group(*update_ops)
        train_step = control_flow_ops.with_dependencies([updates], train_step)

    # Add summaries for BN variables
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cross_entropy', cross_entropy)
    for v in tf.all_variables():
        print(v.name)
        if 'batch_normalization' in v.name:
            tf.summary.histogram(v.name, v)
    merged_summary_op = tf.summary.merge_all()

    return {'x': x,
            'y_': y_,
            'keep_prob': keep_prob,
            'is_training': is_training,
            'train_step': train_step,
            'global_step': step,
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'summary': merged_summary_op}


def train():
    # clear checkpoint directory
    print('Clearing existed checkpoints and logs')
    for root, sub_folder, file_list in os.walk(FLAGS.checkpoint_dir):
        for f in file_list:
            os.remove(os.path.join(root, f))
    for root, sub_folder, file_list in os.walk(FLAGS.train_log_dir):
        for f in file_list:
            os.remove(os.path.join(root, f))

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    net = model()
    sess = tf.Session()

    # DEBUGGING
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir, 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir, 'valid'), sess.graph)

    # Train
    batch_size = FLAGS.batch_size
    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_dict = {net['x']: batch_xs,
                      net['y_']: batch_ys,
                      net['keep_prob']: 0.5,
                      net['is_training']: True}
        step, _ = sess.run([net['global_step'], net['train_step']], feed_dict=train_dict)
        if step % 50 == 0:
            train_dict = {net['x']: batch_xs,
                          net['y_']: batch_ys,
                          net['keep_prob']: 1.0,
                          net['is_training']: True}
            entropy, acc, summary = sess.run([net['cross_entropy'], net['accuracy'], net['summary']],
                                             feed_dict=train_dict)
            train_writer.add_summary(summary, global_step=step)
            print('Train step {}: entropy {}: accuracy {}'.format(step, entropy, acc))

            # Note: the validation error is erratic in the beginning (Maybe 2~3k steps).
            # This does NOT imply the batch normalization is buggy.
            # On the contrary, it's BN's dynamics: moving_mean/variance are not estimated that well in the beginning.
            valid_dict = {net['x']: batch_xs,
                          net['y_']: batch_ys,
                          net['keep_prob']: 1.0,
                          net['is_training']: False}
            entropy, acc, summary = sess.run([net['cross_entropy'], net['accuracy'], net['summary']],
                                             feed_dict=valid_dict)
            valid_writer.add_summary(summary, global_step=step)
            print('***** Valid step {}: entropy {}: accuracy {} *****'.format(step, entropy, acc))
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'mnist-conv-slim'))
    print('Finish training')

    # validation
    acc = 0.0
    batch_size = FLAGS.batch_size
    num_iter = 5000 // batch_size
    for i in range(num_iter):
        batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
        test_dict = {net['x']: batch_xs,
                     net['y_']: batch_ys,
                     net['keep_prob']: 1.0,
                     net['is_training']: False}
        acc_ = sess.run(net['accuracy'], feed_dict=test_dict)
        acc += acc_
    print('Overall validation accuracy {}'.format(acc / num_iter))
    sess.close()


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
    batch_size = FLAGS.batch_size
    num_iter = 10000 // batch_size
    for i in range(num_iter):
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        feed_dict = {net['x']: batch_xs,
                     net['y_']: batch_ys,
                     net['keep_prob']: 1.0,
                     net['is_training']: False}
        acc_ = sess.run(net['accuracy'], feed_dict=feed_dict)
        acc += acc_
    print('Overall test accuracy {}'.format(acc / num_iter))
    sess.close()


def main(_):
    if FLAGS.phase == 'train':
        train()
    else:
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data',
                        help='Directory for storing input data')
    parser.add_argument('--phase', type=str, default='train',
                        help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--train_log_dir', type=str, default='log',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory for checkpoint file')
    FLAGS, unparsed = parser.parse_known_args()
    if not os.path.isdir(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
