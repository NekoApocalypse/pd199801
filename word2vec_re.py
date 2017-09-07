import pickle
import time
import sys
import os

import numpy as np
import tensorflow as tf

class Options(object):
    def __init__(self):
        self.vocab_size = 20000
        self.emb_dim = 20
        self.num_neg_samples = 5
        self.learning_rate = 0.2
        self.batch_size = 20
        self.epochs_to_train = 15
        self.window_size = 5
        self.subsample = 1e-3

        self.stat_interval = 5
        self.summary_interval = 5
        self.checkpoint_interval = 600


class Word2Vec(object):
    def __init__(self, options, session):
        self._session = session
        self._options = options
        self.global_step = 0
        self.build_graph()

    def foward(self, examples, labels):
        """Compute logits from center words and context words"""
        opts = self._options
        init_width = 0.5/opts.emb_dim
        #embeddings layer
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size,opts.emb_dim], -init_width, init_width),
                name='emb'
            )
        self._emb = emb
        #softmax layer
        softmax_w = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name='softmax_w'
        )
        softmax_b = tf.Variable(
            tf.zeros([opts.vocab_size], name='softmax_b')
        )
        #negative sampling
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size,1]
        )
        sample_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=opts.num_samples,
                unique=True,
                range_max=opts.vocab_size,
                distortion=0.75,
                unigrams=opts.vocab_counts.tolost()
            )
        )
        example_emb = tf.nn.embedding_lookup(emb, examples)
        true_w = tf.nn.embedding_lookup(softmax_w, examples)
        true_b = tf.nn.embedding_lookup(softmax_b, examples)

        sampled_w = tf.nn.embedding_lookup(softmax_w, sample_ids)
        sampled_b = tf.nn.embedding_lookup(softmax_b, sample_ids)

        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w),1)+true_b
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True),+sampled_b_vec
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Compute NCE Loss from Logits"""
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def optimize(self, loss):
        """Operation to minimize loss
        training operation assigned as self._train
        """
        opts = self._options
        self.lr = 0.005
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def build_graph(self):
        """Build graph
        """
        opts = self._options
        examples = tf.placeholder(tf.float32, [None, opts.vocab_size],
                                  name='examples')
        labels = tf.placeholder(tf.float32, [None, opts.vocab_size],
                                name='labels')
        true_logits, sampled_logits = self.foward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        self._loss = loss
        self.optimize(loss) #Assign self._train
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self):
        opts = self._options
        now = time.time()
        init_time = now
        for epoch in range(opts.epochs_to_train):
            #self._session.run(self._train, )
            #To do: Feed Dict
            now = time.time()
            print('Epoch %4d, Time used %8ds\n' % (epoch, now-init_time))
            sys.stdout.flush()


def main(_):
    opts = Options()
    with tf.Session() as session:
        model = Word2Vec(opts, session)
        model.train()
        model.saver.save(session,
                         './model',
                         global_step=model.global_step)


if __name__=='__main__':
    tf.app.run()

