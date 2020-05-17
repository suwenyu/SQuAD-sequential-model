import os
import sys
import random
import argparse
import json
import nltk
import numpy as np
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

def total_exs(dataset):
    """
    Returns the total number of (context, question, answer) triples,
    given the data read from the SQuAD json file.
    """
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total

def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens

def write_to_file(out_file, line):
    out_file.write(line + '\n')

def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.
    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)
    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = '' # accumulator
    current_token_idx = 0 # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context): # step through original characters
        if char != u' ' and char != u'\n': # if it's not a space:
            acc += char # add to accumulator
            context_token = str(context_tokens[current_token_idx]) # current word token
            if acc == context_token: # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                acc = '' # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, tier, out_dir):
    num_exs = 0 # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']

        for pid in range(len(article_paragraphs)):
            context = str(article_paragraphs[pid]['context']) # string
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context) # list of strings (lowercase)
            context = context.lower()
            # print(context)

            qas = article_paragraphs[pid]['qas'] # list of questions
            # print(qas)
            
            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)
            # print(charloc2wordloc)
            if charloc2wordloc is None: # there was a problem
                num_mappingprob += len(qas)
                continue # skip this context example
            
            for qn in qas:
                question = str(qn['question']) # string
                question_tokens = tokenize(question)

                ans_text = str(qn['answers'][0]['text']).lower()
                ans_start_charloc = qn['answers'][0]['answer_start']
                ans_end_charloc = ans_start_charloc + len(ans_text)


                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                    num_spanalignprob += 1
                    continue

                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] # answer start word loc
                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1] # answer end word loc
                assert ans_start_wordloc <= ans_end_wordloc

                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]


                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue # skip this question/answer pair

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                num_exs += 1

    print ("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print ("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print ("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print ("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

    # shuffle examples
    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, tier +'.context'), 'w') as context_file,  \
        open(os.path.join(out_dir, tier +'.question'), 'w') as question_file,\
        open(os.path.join(out_dir, tier +'.answer'), 'w') as ans_text_file, \
        open(os.path.join(out_dir, tier +'.span'), 'w') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)


def main():
    train_filename = "./data/train-v1.1.json"
    dev_filename = "./data/dev-v1.1.json"

    train_data = data_from_json(train_filename)
    print("Train data has %i examples total" % total_exs(train_data))

    preprocess_and_write(train_data, 'train', './data/')
    print("Training data has processed")

    dev_data = data_from_json(dev_filename)
    print ("Dev data has %i examples total" % total_exs(dev_data))
    preprocess_and_write(dev_data, 'dev', './data/')

if __name__ == '__main__':
    main()