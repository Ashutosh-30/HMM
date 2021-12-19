
import numpy as np
from hmm import HMM
from collections import defaultdict

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index
    #   - from a tag to its index
    # The order you index the word/tag does not matter,
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    idx = 0
    for w in unique_words:
        if w not in word2idx.keys():
            word2idx[w] = idx
            idx = idx + 1
    for i in range(S):
        tag2idx[tags[i]] = i


    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if
    #   "divided by zero" is encountered, set the entry
    #   to be zero.
    ###################################################

    total_trans_freq = defaultdict(int)
    tag_trans_freq = defaultdict(lambda: defaultdict(int))

    tag_freq_first = defaultdict(int)
    word_pos = defaultdict(lambda: defaultdict(int))

    tag_freq = defaultdict(int)

    for sen in train_data:
        prev_tag = ""
        tag_freq_first[sen.tags[0]] = tag_freq_first[sen.tags[0]] + 1
        for i, word in enumerate(sen.words):
            curr_tag = sen.tags[i]
            tag_freq[curr_tag] += 1
            total_trans_freq[prev_tag] += 1
            word_pos[curr_tag][word] += 1
            tag_trans_freq[prev_tag][curr_tag] += 1
            prev_tag = curr_tag

    for tag in tags:
        idx = tag2idx[tag]
        pi[idx] = tag_freq_first[tag] / len(train_data)
        for word, count in word_pos[tag].items():
            B[idx][word2idx[word]] = count / tag_freq[tag]
        for next_tag in tags:
            A[idx][tag2idx[next_tag]] = tag_trans_freq[tag][next_tag] / total_trans_freq[tag]

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    S = len(tags)
    next_valid_idx = max(model.obs_dict.values()) + 1
    Z = np.full((S, 1), 10 ** -6)
    for sen in test_data:
        for w in sen.words:
            if w not in model.obs_dict.keys():
                model.obs_dict[w] = next_valid_idx
                model.B = np.append(model.B, Z, axis=1)
                next_valid_idx += 1
        tagging.append(model.viterbi(sen.words))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words


