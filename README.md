# mnist-bn
Using slim to perform batch normalization

Run `python mnist_bn --phase=train` to train.
Run `python mnist_bn --phase=train` to test.

Now I know **the keys to use batch normalization in `slim`**:

1. **Do not hurry.** Because BN uses moving average to estimate population mean/variance, **it takes sometime to "warm up"**. So for the first few batches, even if training accuracy is high, validation accuracy maybe erratic. This is normal: just wait for running mean/variance to warm up.
2. **Use `slim.learning.create_train_op` to create train op instead of `tf.train.GradientDescentOptimizer(0.1).minimize(loss)` or something else!**.
