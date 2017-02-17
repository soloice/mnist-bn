# mnist-bn
Using slim to perform batch normalization

This code works fine. But when I try to use `tf.summary.histogram()` to record distributions of BN paramters, its performance gets really poor. I have no idea about this. Is it a bug with `tf.summary` ?
