import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf

class BasicAttn(tf.keras.Model):
    def __init__(self, dropout_rate, key_vec_size, value_vec_size):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def call(self, values, values_mask, keys):
        values_t = tf.transpose(values, perm=[0, 2, 1])
        attn_logits = tf.matmul(keys, values_t) 

        attn_logits_mask = tf.expand_dims(values_mask, 1)
        _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2)

        # attn_dist = tf.nn.softmax(attn_logits, 2)
        output = tf.matmul(attn_dist, values)

        output = tf.nn.dropout(output, self.dropout_rate)

        return attn_dist, output


class BidafAttn(tf.keras.Model):
    def __init__(self, dropout_rate, vec_size):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.vec_size = vec_size
        self.fully_connected_layer = tf.keras.layers.Dense(1)


    def call(self, q, q_mask, c, c_mask):
        c_expand = tf.expand_dims(c, 2)
        q_expand = tf.expand_dims(q, 1)

        c_pointWise_q = c_expand * q_expand
        # print(c_pointWise_q.shape)
        c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1])
        q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])


        concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1)
        # print(concat_input.shape)

        similarity = tf.reduce_sum(self.fully_connected_layer(concat_input), axis=3)
        similarity_mask = tf.expand_dims(q_mask, 1)

        _, c2q_dist = masked_softmax(similarity, similarity_mask, 2)
        # c2q_dist = tf.nn.softmax(similarity, 2)
        c2q = tf.matmul(c2q_dist, q)
        # print(c2q.shape)

        S_max = tf.reduce_max(similarity, axis=2)

        _, c_dash_dist = masked_softmax(S_max, c_mask, 1)
        # c_dash_dist = tf.nn.softmax(S_max, 1)
        # print(c_dash_dist.shape)

        c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1)
        c_dash = tf.matmul(c_dash_dist_expand, c)

        c_c2q = c * c2q
        c_c_dash = c * c_dash


        output = tf.concat([c2q, c_c2q, c_c_dash], axis=2)
        output = tf.nn.dropout(output, self.dropout_rate)
        
        return output


def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)
    masked_logits = tf.add(logits, exp_mask)
    prob_dist = tf.nn.softmax(masked_logits, dim)

    return masked_logits, prob_dist




