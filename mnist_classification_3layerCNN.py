import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)
print('mnist.train.num_examples = ', mnist.train.num_examples)
print('mnist.test.labels = ', len(mnist.test.labels))
batch_size = 100
n_batch = mnist.train.num_examples//batch_size


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


#ksize 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
#因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#strides 和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	
def max_pool_2x2_s1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

#输入图片shape:28x28x1
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#下面是模型结构
x_image = tf.reshape(x, [-1, 28, 28, 1])
#第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])  #[filter_height, filter_width, in_channels, out_channels]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第三个卷积层
W_conv3 = weight_variable([3,3,64,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
h_pool3 = max_pool_2x2_s1(h_conv3)


#7*7*64输入的神经元的个数（64个7*7的小图片），做一个1024的全连接
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#dropout在这里
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第二个全连接层，转为10维
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#交叉熵损失，这是softmax的损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#自适应动量优化方法（本质是SGD） adaptive moment optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Start Trainging...')
start_time = time.clock()
#tensorflow 必须在session中训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        print('epoch', epoch)
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1})
        str_test = 'epoch = %d, accuracy in test = %.4f' % (epoch, acc_test)
        print(str_test)

end_time = time.clock()
print("Used time: {:.2f} minutes".format((end_time-start_time)/60))