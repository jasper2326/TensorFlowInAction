import tensorflow as tf


global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

