import os
import io
import json
import sys
import logging
import time

from collections import Counter
import string
import re

import tensorflow as tf
import numpy as np

from qa_model import QAModel
from data_batcher import get_batch_generator

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(model, word2id, FLAGS, dev_context_path, dev_qn_path, dev_ans_path):
    logging.info("Calculating F1/EM for all examples in dev set...")

    f1_total = 0.
    em_total = 0.
    example_num = 0

    tic = time.time()

    for batch in get_batch_generator(word2id, dev_context_path, dev_qn_path, dev_ans_path, FLAGS.batch_size, context_len=FLAGS.context_len, question_len=FLAGS.question_len, discard_long=False):
        # print(type(batch))
        prob_start, prob_end = model.predict([batch.context_ids, batch.context_mask, batch.qn_ids, batch.qn_mask])
        
        start_pos = np.argmax(prob_start, axis=1)
        end_pos = np.argmax(prob_end, axis=1)

        pred_start_pos = start_pos.tolist()
        pred_end_pos = end_pos.tolist()

        for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
            example_num += 1

            # Get the predicted answer
            # Important: batch.context_tokens contains the original words (no UNKs)
            # You need to use the original no-UNK version when measuring F1/EM
            pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
            pred_answer = " ".join(pred_ans_tokens)


            # Get true answer (no UNKs)
            true_answer = " ".join(true_ans_tokens)

            # Calc F1/EM
            f1 = f1_score(pred_answer, true_answer)
            em = exact_match_score(pred_answer, true_answer)
            f1_total += f1
            em_total += em

            # print(f1, em, example_num)


    f1_total /= example_num
    em_total /= example_num

    toc = time.time()
    logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

    return f1_total, em_total



