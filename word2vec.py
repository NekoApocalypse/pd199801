import pickle
import time
import sys
import os

import numpy as np
import tensorflow as tf


class Options(object):
   '''
   Options used by Word2Vec Model
   '''
   def __init__(self):
       self.vocab_size = 0
       self.emb_dim = 40
       self.num_neg_samples = 5
       self.learning_rate = 0.2
       self.batch_size = 20
       self.epochs_to_train = 15
       self.concurrent_steps = 12
       self.window_size = 5
       self.min_count = 5
       self.subsample = 1e-3

       self.stat_interval = 5
       self.summary_interval = 5
       self.checkpoint_interval = 600
       #Counteed in Seconds


class Word2Vec(object):
    def __init__(self, options, session):
        self._session = session
        self._options = options
        self.build_graph()


    def forward(self, examples, labels):
        opts = self._options
        init_width = 0.5/opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size,opts.emb_dim],-init_width,init_width),
                name='emb'
            )
        self._emb = emb
        softmax_w = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="softmax_w"
        )
        softmax_b = tf.Variable(
            tf.zeros([opts.voab_size], name='softmax_b')
        )
        self.global_step = tf.Variable(0, name='global_step')

        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype = tf.int64),
            [opts.batch_size,1]
        )

        sample_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=opts.num_samples,
                unique=True,
                range_max=opts.vocab_size,
                distortion=0.75,
                unigrams=opts.vocab_counts.tolist()
            )
        )
        example_emb = tf.nn.embedding_lookup(emb, examples)
        true_w = tf.nn.embedding_lookup(softmax_w, examples)
        true_b = tf.nn.embedding_lookup(softmax_b, examples)

        sampled_w = tf.nn.embedding_lookup(softmax_w, sample_ids)
        sampled_b = tf.nn.embedding_lookup(softmax_b, sample_ids)

        true_logits = tf.reduce_sum(tf.multiply(example_emb,true_w),1)+true_b

        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True),+sampled_b_vec
        return true_logits, sampled_logits


    def nce_loss(self, true_logits, sampled_logits):
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits),logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits),logits=sampled_logits)

        nce_loss_tensor = (tf.reduce_sum(true_xent)+tf.reduce_sum(sampled_xent) / opts.batch_size)
        return nce_loss_tensor


    def optimize(self,loss):
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        #lr = opts.learning_rate * tf.maximum(0.0001, 1.0-tf.cast(self._words, tf.float32)/ words_to_train)
        #self.lr = lr
        self.lr = 0.005
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train


    def build_graph(self):
        opts = self._options
        '''
        (words, counts, words_per_epoch, self._epoch, self._words, examples, labels) =
            word2vec.skipgram_word2vec(filename=opts.train_data,
                                        batch_size=opts.batch_size,
                                        window_size=opts.window_size,
                                        min_count=opts.min_count,
                                        subsample=opts.subsample)
        '''
        examples = tf.placeholder(tf.float32, [None, opts.vocab_size], name='examples')
        labels = tf.placeholder(tf.float32, [None, opts.vocab_size], name='labels')
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NEC loss",loss)
        self._loss = loss
        self.optimize(loss)     #Define self._train

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


    def train(self):
        opts = self._options

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)

        #last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_time, last_summary_time = time.time(), 0
        last_checkpoint_time = 0

        for epoch in range(opts.epochs_to_train):
            self._session.run(self.train, )     #pending: feed_dict
            now = time.time()
            print('Epoch %4d' % epoch)


        while True:
            time.sleep(opts.statistics_interval)
            '''
            (epoch, step, loss, words, lr) = self._session.run(
                [self._epoch, self.global_step, self._loss, self._words, self.lr]
            )
            '''
            self._session.run(self._train, )    # pending: feed_dict
            now = time.time()
            last_words, last_time, rate = words, now, (words-last_words) / (now-last_time)
            print('Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r' %
                  (epoch, step, lr, loss, rate), end='')
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                #summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(summary_str)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session, os.path.join(opts.save_path, 'model.ckpt'),
                                global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != initial_epoch:
                break


def main(_):
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Word2Vec(opts, session)
        for _ in range(opts.epochs_to_train):
            model.train()
        model.saver.save(session,
                         os.path.join(opts.save_path,'model.ckpy'),
                         glabal_step=model.global_step)

if __name__ =='__main__':
    tf.app.run()
