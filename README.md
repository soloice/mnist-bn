# mnist-bn
Using slim to perform batch normalization

Run `python mnist_bn.py --phase=train` to train.
Run `python mnist_bn.py --phase=test` to test.

It should achieve an accuracy of ~99.3% or higher on test set.


**The keys to use batch normalization in `slim`** are:

1. **Set proper decay rate for BN layer.** Because a BN layer uses EMA (exponential moving average) to approximate the population mean/variance, it takes sometime to warm up, i.e. to get the EMA close to real population mean/variance. The default decay rate is 0.999, which is kind of high for our little cute MNIST dataset and needs ~1000 steps to get a good estimation. In my code, `decay` is set to 0.95, then it learns the population statistics very quickly.  However, a large value of `decay` does have it own advantage: it gathers information from more mini-batches thus is more stable.

2. **Use `slim.learning.create_train_op` to create train op instead of `tf.train.GradientDescentOptimizer(0.1).minimize(loss)` or something else!**.

I've added accuracy, cross_entropy and batch normalization paramters into summary. Use **tensorboard --logdir=/log** to explore the learning curve and parameter distributions!
