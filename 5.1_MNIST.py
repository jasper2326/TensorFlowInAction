from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/Users/jasper/Desktop/test/mnist', one_hot=True)

#print ('Training data size: ', mnist.train.num_examples)
#print ('Validating data size: ', mnist.validation.num_examples)
#print ('Testing data size: ', mnist.test.num_examples)
#print ('Example training data: ', mnist.train.images[0])
#print (mnist.train.labels[0])


batch_size = 100
# x is input image & y is the label
xs, ys = mnist.train.next_batch(batch_size)
print (xs.shape, ys.shape)

