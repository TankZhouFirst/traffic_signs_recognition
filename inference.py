import tensorflow as tf

def init_weights(shape, lambd):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

def init_bias(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(images, keep_prob,lambd):
    # 第一层
    # 初始化参数
    W_conv1 = init_weights([5, 5, 1, 64],lambd)
    b_conv1 = init_bias([1, 64])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层
    W_conv2 = init_weights([5, 5, 64, 128],lambd)
    b_conv2 = init_bias([1, 128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层1
    W_fc1 = init_weights(shape=[7 * 7 * 128, 1024], lambd=lambd)
    b_fc1 = init_bias(shape=[1, 1024])
    h_pool2_flatten = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1) + b_fc1)

    # dropout layer
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = init_weights(shape=[1024, 62],lambd=lambd)
    b_fc2 = init_bias(shape=[1, 62])
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return logits