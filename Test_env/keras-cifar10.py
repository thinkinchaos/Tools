import timeit
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data


def model():
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    rate = tf.placeholder(tf.float32)
    # convolutional layer 1
    conv_1 = tf.layers.conv2d(x, 32, [3, 3], padding='SAME', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling2d(conv_1, [2, 2], strides=2, padding='SAME')
    drop_1 = tf.layers.dropout(max_pool_1, rate=rate)
    # convolutional layer 2
    conv_2 = tf.layers.conv2d(drop_1, 64, [3, 3], padding="SAME", activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling2d(conv_2, [2, 2], strides=2, padding="SAME")
    drop_2 = tf.layers.dropout(max_pool_2, rate=rate)
    # convolutional layers 3
    conv_3 = tf.layers.conv2d(drop_2, 128, [3, 3], padding="SAME", activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling2d(conv_3, [2, 2], strides=2, padding="SAME")
    drop_3 = tf.layers.dropout(max_pool_3, rate=rate)
    # fully connected layer 1
    flat = tf.reshape(drop_3, shape=[-1, 4 * 4 * 128])
    fc_1 = tf.layers.dense(flat, 80, activation=tf.nn.relu)
    drop_4 = tf.layers.dropout(fc_1 , rate=rate)
    # fully connected layer 2 or the output layers
    fc_2 = tf.layers.dense(drop_4, 10)
    output = tf.nn.relu(fc_2)
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
    return x, y, rate, accuracy, loss, optimizer


def one_hot_encoder(y):
    ret = np.zeros(len(y) * 10)
    ret = ret.reshape([-1, 10])
    for i in range(len(y)):
        ret[i][y[i]] = 1
    return (ret)


def train(data_train, label_train, sess, x, y, rate, optimizer, accuracy, loss):
    batch_size = 128
    label_train_cls = one_hot_encoder(label_train)
    start = end = 0
    for i in range(int(len(data_train) / batch_size)):
        if (i + 1) % 100 == 1:
            start = timeit.default_timer()
        batch_x = data_train[i * batch_size:(i + 1) * batch_size]
        batch_y = label_train_cls[i * batch_size:(i + 1) * batch_size]
        _, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={x:batch_x, y:batch_y, rate:0.4})
        if (i + 1) % 100 == 0:
            end = timeit.default_timer()
            print("Time:", end-start, "s the loss is ", batch_loss, " and the accuracy is ", batch_accuracy * 100, "%")


def test(data_test, label_test, sess, x, y, rate, accuracy, loss):
    batch_size = 64
    label_test_cls = one_hot_encoder(label_test)
    global_loss = 0
    global_accuracy = 0
    for t in range(int(len(data_test) / batch_size)):
        batch_x = data_test[t * batch_size : (t + 1) * batch_size]
        batch_y = label_test_cls[t * batch_size : (t + 1) * batch_size]
        batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={x:batch_x, y:batch_y, rate:1})
        global_loss += batch_loss
        global_accuracy += batch_accuracy
    global_loss = global_loss / (len(data_test) / batch_size)
    global_accuracy = global_accuracy / (len(data_test) / batch_size)
    print("In test-env Time, loss is ", global_loss, ' and the accuracy is ', global_accuracy)


EPOCH = 100
(data_train, label_train), (data_test, label_test) = load_data()
print("There is ", len(data_train), " training images and ", len(data_test), " images")
x, y, rate, accuracy, loss, optimizer = model()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(EPOCH):
    print("Train on epoch ", i ," start")
    train(data_train, label_train, sess, x, y, rate, optimizer, accuracy, loss)
    test(data_train, label_train, sess, x, y, rate, accuracy, loss)