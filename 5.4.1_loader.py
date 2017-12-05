import tensorflow as tf


saver = tf.train.import_meta_graph('/Users/jasper/Desktop/test/TensorFlowSaver/model.ckpt/model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, '/Users/jasper/Desktop/test/TensorFlowSaver/model.ckpt')
    print sess.run(tf.get_default_graph().get_tensor_by_name('add:0'))