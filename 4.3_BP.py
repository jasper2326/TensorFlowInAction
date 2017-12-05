import tensorflow as tf


batch_size = 8

x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y_input')

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    STEPS = 10000
    for i in range(STEPS):
        pass
