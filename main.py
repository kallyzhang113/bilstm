# coding: utf-8
import tensorflow as tf
import random
import numpy as np
import json
from sklearn import metrics

class GenerateData(object):
    def __init__(self, users, scaler=None):
        print "GENERATING DATA"
        self.data = []
        self.labels = []
        self.seqlen = []
        self.profile = []
        self.max_seq_len = 20

        for user in users:
            commit_seq = user['event_seq'][-min(self.max_seq_len, len(user['event_seq'])):]
            self.seqlen.append(len(commit_seq))
            self.data.append(commit_seq)
            self.profile.append(user['profile'])
            if user['label'] == 1:
                self.labels.append([0., 1.])
            else:
                self.labels.append([1., 0.])

        self.scaler = self.pre_processing(scaler)
        self.feature_len = len(users[0]['event_seq'][0])
        self.batch_id = 0

        for i in range(len(self.data)):
            self.data[i] += [[0.] * self.feature_len for _ in range(self.max_seq_len - self.seqlen[i])]

        print len(self.data)

    def pre_processing(self, scaler):
        from sklearn import preprocessing
        print "START NORMALIZATION"
        seq_list = []
        for user in self.data:
            for seq in user:
                seq_list.append(seq)

        if not scaler:
            scaler = preprocessing.MinMaxScaler().fit(seq_list)
        seq_list = scaler.transform(seq_list)

        start = 0
        data_normalized = []
        for i in range(len(self.data)):
            data_normalized.append(seq_list[start: (start + self.seqlen[i])].tolist())
            start += self.seqlen[i]

        self.data = data_normalized
        return scaler

    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

def get_a_cell(n_hidden, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return drop

def dynamicRNN(x, seqlen, weights, biases, n_hidden, keep_prob, seq_max_len):

    lstm_cell_fw = get_a_cell(n_hidden[0], keep_prob[0])
    lstm_cell_bw = get_a_cell(n_hidden[1], keep_prob[1])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=x, dtype=tf.float32, sequence_length=seqlen)

    out = tf.concat((states[0][-1], states[1][-1]), axis=1)

    return tf.matmul(out, weights['out']) + biases['out']

def detect(training_users, testing_users):
    learning_rate = 0.01
    training_iters = 500000
    batch_size = 128
    display_step = 10

    trainset = GenerateData(training_users)
    testset = GenerateData(testing_users, trainset.scaler)

    seq_max_len = trainset.max_seq_len
    feature_len = trainset.feature_len
    n_hidden = [16, 16]
    n_classes = 2
    train_keep_prob = [1., 1.]

    x = tf.placeholder("float", [None, seq_max_len, feature_len])
    y = tf.placeholder("float", [None, n_classes])
    seqlen = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.int32)

    weights = {
        'out': tf.Variable(tf.random_normal(shape=[sum(n_hidden), n_classes], stddev=0.4))
    }
    biases = {
        'out': tf.Variable(tf.zeros(shape=[n_classes]))
    }

    pred = dynamicRNN(x, seqlen, weights, biases, n_hidden, train_keep_prob, seq_max_len)
    cost0 = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    cost = tf.reduce_mean(cost0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    g = tf.gradients(cost, [weights['out'], biases['out']])

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, keep_prob: train_keep_prob})

            if step % display_step == 0:
                y_p = tf.argmax(pred, 1)
                acc, y_pred, loss = sess.run([accuracy, y_p, cost],
                               feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, keep_prob: train_keep_prob})
                y_true = np.argmax(batch_y, 1)
                print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) +
                      ", Training Accuracy = " + "{:.5f}".format(acc) +
                      ", Training Precision = " + "{:.5f}".format(metrics.precision_score(y_true, y_pred)) +
                      ", Training Recall = " + "{:.5f}".format(metrics.recall_score(y_true, y_pred)) +
                      ", Training f1 score = " + "{:.5f}".format(metrics.f1_score(y_true, y_pred)))
            step += 1
        print("Optimization Finished!")

        saver = tf.train.Saver()
        saver.save(sess, "model/model")

        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        y_p = tf.argmax(pred, 1)
        y_true = np.argmax(test_label, 1)
        acc, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_data, y: test_label, seqlen: test_seqlen, keep_prob: 1.0})
        print("Testing Accuracy:", acc)
        print("Testing AUC:", metrics.roc_auc_score(y_true, y_pred))
        print metrics.classification_report(y_true, y_pred, digits=4)
