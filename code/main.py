import os
import io
import json
import sys
import logging
import time

from argparse import ArgumentParser

import tensorflow as tf
import numpy as np

from qa_model import QAModel
from load_pretrained import get_glove
from data_batcher import get_batch_generator
from eval import evaluate

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

# High-level options
parser = ArgumentParser()
parser.add_argument("--gpu", help="Which GPU to use, if you have multiple.", default=0, dest="gpu")
parser.add_argument("--mode", help="Available modes: train / show_examples / official_eval", default="train", dest="mode")
parser.add_argument("--experiment_name", help="Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment", default="", dest="experiment_name")
parser.add_argument("--num_epochs", help="Number of epochs to train. 0 means train indefinitely", default=0, dest="num_epochs", type=int)

# Hyperparameters
parser.add_argument("--learning_rate", help="Learning rate.", default=0.001, dest="learning_rate", type=float)
parser.add_argument("--max_gradient_norm", help="Clip gradients to this norm.", default=5.0, dest="max_gradient_norm")
parser.add_argument("--dropout", help="Fraction of units randomly dropped on non-recurrent connections.", default=0.15, dest="dropout", type=float)
parser.add_argument("--batch_size", help="Batch size to use", default=60, dest="batch_size", type=int)
parser.add_argument("--hidden_size_encoder", help="Size of the hidden states", default=150, dest="hidden_size_encoder", type=int)
parser.add_argument("--hidden_size_qp_matching", help="Size of the hidden states", default=150, dest="hidden_size_qp_matching", type=int)
parser.add_argument("--hidden_size_sm_matching", help="Size of the hidden states", default=50, dest="hidden_size_sm_matching", type=int)
parser.add_argument("--hidden_size_fully_connected", help="Size of the hidden states", default=200, dest="hidden_size_fully_connected", type=int)
parser.add_argument("--context_len", dest="context_len", default=300, help="The maximum context length of your model", type=int)
parser.add_argument("--question_len", dest="question_len", default=30, help="The maximum question length of your", type=int)
parser.add_argument("--embedding_size", dest="embedding_size", default=100, help="Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300", type=int)

# Bool FLAGS to select different models
parser.add_argument("--do_char_embed", dest="do_char_embed", help="Include char embedding -True/False", action='store_true')
parser.add_argument("--add_highway_layer", dest="add_highway_layer", help="Add highway layer to concatenated embeddings -True/False", action='store_true')
parser.add_argument("--cnn_encoder", dest="cnn_encoder", help="Add CNN Encoder Layer -True/False", action='store_true')
parser.add_argument("--rnet_attention", dest="rnet_attention", help="Perform RNET QP and SM attention-True/False", action='store_true')
parser.add_argument("--bidaf_attention", dest="bidaf_attention", help="Use BIDAF Attention-True/False", action='store_true')
parser.add_argument("--answer_pointer_RNET", dest="answer_pointer_RNET", help="Use Answer Pointer from RNET-True/False", action='store_true')
parser.add_argument("--smart_span", dest="smart_span", help="Select start and end idx based on smart conditions-True/False", action='store_true')


# Hyperparameters for BIDAF
parser.add_argument("--hidden_size_modeling", dest="hidden_size_modeling", default=150, help="Size of modeling layer", type=int)

# How often to print, save, eval
parser.add_argument("--print_every", dest="print_every", default=1, help="How many iterations to do per print.", type=int)
parser.add_argument("--save_every", dest="save_every", default=500, help="How many iterations to do per save.", type=int)
parser.add_argument("--eval_every", dest="eval_every", default=500, help="How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.", type=int)
parser.add_argument("--keep", dest="keep", default=1, help="How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).", type=int)

# Reading and saving data
parser.add_argument("--train_dir", dest="train_dir", default="", help="Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
parser.add_argument("--glove_path", dest="glove_path", default="", help="Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
parser.add_argument("--data_dir", dest="data_dir", default=DEFAULT_DATA_DIR, help="Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
parser.add_argument("--ckpt_load_dir", dest="ckpt_load_dir", default="", help="For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
parser.add_argument("--json_in_path", dest="json_in_path", default="", help="For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
parser.add_argument("--json_out_path", dest="json_out_path", default="predictions.json", help="Output path for official_eval mode. Defaults to predictions.json")


FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

# print(FLAGS)

def main():
    print ("Your TensorFlow version: %s" % tf.__version__)

    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR + "/glove.6B/", "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

    
    global_step = 1
    epoch = 0
    print("Beginning training loop...")

    # Initialize model
    model = QAModel(FLAGS, id2word, word2id, emb_matrix)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    
    while FLAGS.num_epochs == 0 or epoch < FLAGS.num_epochs:
        epoch += 1
        epoch_tic = time.time()
        
        for batch in get_batch_generator( \
            word2id, train_context_path, train_qn_path, \
            train_ans_path, FLAGS.batch_size, context_len=FLAGS.context_len, \
            question_len=FLAGS.question_len, discard_long=True):
            # print(batch.ans_span)
            
            with tf.GradientTape() as tape:
                prob_start, prob_end = model([batch.context_ids, batch.context_mask, batch.qn_ids, batch.qn_mask])
                # prob_start, prob_end = model(batch.context_ids, batch.context_mask, batch.qn_ids, batch.qn_mask)
                
                loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob_start, labels=batch.ans_span[:, 0])
                loss_start = tf.reduce_mean(loss_start)

                loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob_end, labels=batch.ans_span[:, 1])
                loss_end = tf.reduce_mean(loss_end)

                loss = loss_start + loss_end
                # print("loss %f" % (loss.numpy()))
            
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

            if global_step % FLAGS.eval_every == 0 :
                print("==== start evaluating ==== ")
                dev_f1, dev_em = evaluate(model, word2id, FLAGS, dev_context_path, dev_qn_path, dev_ans_path)
                print("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                print("==========================")
            global_step += 1

        epoch_toc = time.time()
        print("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

    sys.stdout.flush()


if __name__ == "__main__":
    main()
