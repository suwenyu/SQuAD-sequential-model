import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops.rnn_cell import DropoutWrapper

from attention import BasicAttn, BidafAttn, masked_softmax

logging.basicConfig(level=logging.INFO)

class SimpleSoftmaxLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fully_connected_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        inp = inputs[0]
        mask = inputs[1]

        logits = self.fully_connected_layer(inp)
        logits = tf.squeeze(logits, axis=[2])

        masked_logits, prob_dist = masked_softmax(logits, mask, 1)
        # prob_dist = tf.nn.softmax(logits, 1)
        return masked_logits, prob_dist


class QAModel(tf.keras.Model):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        super().__init__()

        print ("Initializing the QAModel...")
        self.FLAGS = FLAGS
        self.keep_prob = 1.0 - self.FLAGS.dropout

        self.embeddings = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.FLAGS.hidden_size_encoder, return_sequences=True), merge_mode="concat")

        self.fully_connected_layer = tf.keras.layers.Dense(self.FLAGS.hidden_size_fully_connected)

        self.softmax_layer_start = SimpleSoftmaxLayer()
        self.softmax_layer_end = SimpleSoftmaxLayer()

        self.bilstm_attn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.FLAGS.hidden_size_modeling, return_sequences=True), merge_mode="concat")

    def call(self, inputs):
    # def call(self, context_inp, context_mask, qn_inp, qn_mask):
        context_inp = inputs[0]
        qn_inp = inputs[2]
        context_mask = inputs[1]
        qn_mask = inputs[3]

        self.context_emb = embedding_ops.embedding_lookup(self.embeddings, context_inp)
        self.qn_emb = embedding_ops.embedding_lookup(self.embeddings, qn_inp)
        
        # context_hiddens = self.biLSTM(self.context_emb, context_mask)
        # question_hiddens = self.biLSTM(self.qn_emb, qn_mask)

        context_hiddens = self.bilstm(self.context_emb)
        question_hiddens = self.bilstm(self.qn_emb)
        # print(context_hiddens.shape) batch_size * ctx_len * h*2
        
        last_dim = context_hiddens.get_shape().as_list()[-1]

        # attention
        if self.FLAGS.bidaf_attention:
            self.attn = BidafAttn(self.FLAGS.dropout, self.FLAGS.hidden_size_encoder * 2)
            attn_output = self.attn(question_hiddens, qn_mask, context_hiddens, context_mask)
            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)

            attention_hidden = self.bilstm_attn(blended_reps)
            blended_reps = attention_hidden


        else:
            self.attn = BasicAttn(self.FLAGS.dropout, last_dim, last_dim)
            _, attn_output = self.attn(question_hiddens, qn_mask, context_hiddens)


            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)
            # print(blended_reps.shape)
        
        blended_reps_final = self.fully_connected_layer(blended_reps)
        
        logits_start, probdist_start = self.softmax_layer_start([blended_reps_final, context_mask])
        logits_end, probdist_end = self.softmax_layer_end([blended_reps_final, context_mask])

        return logits_start, logits_end
