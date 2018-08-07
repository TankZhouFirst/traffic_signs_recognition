from get_input import get_input
from inference import inference
import tensorflow as tf

TRAIN_DATA_PATH = './dataset/Training'
TEST_DATA_PATH = './dataset/Testing'

def train(lambd):
    # 首先读取所有数据
    train_images,train_labels = get_input(TRAIN_DATA_PATH)
    test_images, test_labels = get_input(TEST_DATA_PATH)

    x_inputs = tf.placeholder(tf.float32, shape=[None, 28,28])
    y_labels = tf.placeholder(tf.float32, shape=[None, 62])
    x_imgs = tf.reshape(x_inputs, [-1,28,28,1])
    y_labs = tf.cast(y_labels, tf.float32)

    keep_prob = tf.placeholder(tf.float32)

    logits = inference(x_imgs, keep_prob,lambd)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labs, logits=logits))
    tf.add_to_collection('losses', cross_entropy)
    loss_op = tf.add_n(tf.get_collection('losses'))

    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_labs, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            index = 0
            while index < len(train_labels):
                if index + 50 > len(train_labels):
                    xs = train_images[index:]
                    ys = train_labels[index:]
                else:
                    xs = train_images[index:index+50]
                    ys = train_labels[index:index+50]
                index += 50
                _, loss= sess.run([train_op, loss_op],feed_dict={x_inputs:xs, y_labels:ys, keep_prob:0.3})
                print('第 %d 次迭代，loss = %.3f' % (i, loss))

            xs = train_images
            ys = train_labels
            train_acc = sess.run(accuracy_op, feed_dict={x_inputs: xs, y_labels: ys, keep_prob: 1.0})
            print('训练集精度为：%.1f %%' % (train_acc * 100))
            xs = test_images
            ys = test_labels
            test_acc = sess.run(accuracy_op, feed_dict={x_inputs: xs, y_labels: ys, keep_prob: 1.0})
            print('测试集精度为：%.1f %%' % (test_acc * 100))

train(0.02)