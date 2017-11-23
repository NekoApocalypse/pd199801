import pickle
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data_loader as dl

flags = tf.app.flags

class Options(object):
    def __init__(self):
        # assigned externally
        self.vocab_size = 0
        self.emb_dim = 20
        self.num_neg_samples = 5
        self.learning_rate = 0.1
        self.batch_size = 16
        self.epochs_to_train = 5
        self.window_size = 5
        self.subsample = 1e-3
        self.stat_interval = 5
        self.summary_interval = 5
        self.checkpoint_interval = 500
        self.restore_checkpoint = False
        self.save_path = './checkpoints/'


class Word2Vec(object):
    def __init__(self, options, session, restore=False):
        self._session = session
        self._options = options
        self.lr = options.learning_rate
        self.book = dl.DailyVocabulary(
            window=options.window_size,
            batch_size=options.batch_size)
        self._options.vocab_size = len(self.book.id2word)
        self.loss_summary = []
        # defined in self.build_graph() & self.restore_graph()
        # All variables below are saved by saver
        # self.saver
        # self._examples            : Placeholder for examples (center words)
        # self._labels              : Placeholder for labels (context words)
        # self._emb                 : Embedding matrix
        # self._loss                : NCE loss
        # self._train               : Operation to minimize self._loss
        # self.global_step_tensor   : Global step tracker
        if not restore:
            self.build_graph()
        else:
            self.restore_graph()

    def forward(self, examples, labels):
        """ Compute true_logits and sampled_logits from input """
        opts = self._options
        # embeddings layer
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name='emb'
        )
        # softmax layer
        softmax_w = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name='softmax_w'
        )
        softmax_b = tf.Variable(
            tf.zeros([opts.vocab_size], name='softmax_b')
        )
        # negative sampling
        labels_matrix = tf.reshape(
            tf.cast(labels, dtype=tf.int64),
            [opts.batch_size, 1]
        )
        sample_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_neg_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=self.book.id2count
        ))
        # [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)
        # [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(softmax_w, labels)
        # [batch_size, 1]
        true_b = tf.nn.embedding_lookup(softmax_b, labels)
        # [num_neg_samples, emb_dim]
        sampled_w = tf.nn.embedding_lookup(softmax_w, sample_ids)
        # [num_neg_samples, 1]
        sampled_b = tf.nn.embedding_lookup(softmax_b, sample_ids)
        '''
        print('Shape of softmax_w')
        print(softmax_w.get_shape())
        print('Shape of true_w')
        print(true_w.get_shape())
        print('Shape of true_b')
        print(true_b.get_shape())
        print('Shape of example_emb')
        print(example_emb.get_shape())
        '''
        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
        # Sampled logits: [batch_size, num_neg_samples]
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_neg_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits, emb

    def nce_loss(self, true_logits, sampled_logits, name=None):
        """ Compute NCE Loss from Logits """
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
        nce_loss_tensor = tf.divide(
            tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent),
            tf.cast(opts.batch_size, tf.float32),
            name=name
        )
        return nce_loss_tensor

    def optimize(self, loss):
        """
        Operation to minimize loss
        training operation assigned as self._train
        """
        opts = self._options
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        train = optimizer.minimize(loss,
                                   global_step=global_step_tensor,
                                   gate_gradients=optimizer.GATE_NONE,
                                   name='train')
        return train, global_step_tensor

    def build_graph(self):
        """ Build graph """
        opts = self._options
        examples = tf.placeholder(tf.int32, [None],
                                  name='examples')
        labels = tf.placeholder(tf.int32, [None],
                                name='labels')
        true_logits, sampled_logits, emb = self.forward(examples, labels)  # defines self._emb
        self._emb = emb
        self._loss = self.nce_loss(true_logits, sampled_logits, name='loss')
        self._train, self.global_step_tensor = self.optimize(self._loss)
        self._session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._examples = examples
        self._labels = labels

    def restore_graph(self):
        self.saver = tf.train.import_meta_graph('./checkpoints/model-560720.meta')
        self.saver.restore(self._session, tf.train.latest_checkpoint('./checkpoints'))
        graph = tf.get_default_graph()
        self.global_step_tensor = graph.get_tensor_by_name('global_step:0')
        self._emb = graph.get_tensor_by_name('emb:0')
        # self._train = graph.get_tensor_by_name('train:0')
        # self._train = graph.get_operation_by_name('train')
        self._loss = graph.get_tensor_by_name('loss:0')
        self._labels = graph.get_tensor_by_name('labels:0')
        self._examples = graph.get_tensor_by_name('examples:0')

    def train(self):
        opts = self._options
        now = time.time()
        init_time = now
        stat_time = now
        print('training started')
        for epoch in range(opts.epochs_to_train):
            self.book.restart_epoch()
            while True:
                examples, labels = self.book.generate_batch()
                # if len(examples) < 40:
                if len(examples) < opts.batch_size:
                    break
                test_loss, _ = self._session.run([self._loss, self._train],
                        feed_dict={self._examples: examples, self._labels: labels})
                self.loss_summary.append(test_loss)
                now = time.time()
                if now - stat_time > opts.stat_interval:
                    stat_time = now
                    print('Epoch %4d, Global Step %4d, Time used %8ds' % (
                        epoch, self.global_step_tensor.eval(), now - init_time))
                    print('Loss: ', test_loss)
                    sys.stdout.flush()
                if self.book.end_of_epoch:
                    break


def main(_):
    opts = Options()
    with tf.Session() as session:
        model = Word2Vec(opts, session, opts.restore_checkpoint)
        if not opts.restore_checkpoint:
            model.train()
            model.saver.save(session,
                             './checkpoints/model',
                             global_step=model.global_step_tensor)
            with open('loss_summary.dat', 'wb') as f:
                pickle.dump(model.loss_summary, f)
        else:
            with open('loss_summary.dat', 'rb') as f:
                model.loss_summary = pickle.load(f)
            print('Model successfully restored, global steps = %d\n' %
                  model.global_step_tensor.eval())
    plt.plot(model.loss_summary)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
