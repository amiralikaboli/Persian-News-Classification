import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from train import CharRNN


class LanguageModel:
    def __init__(self, checkpoint_path, hyper_parameters):
        with open('data/index2char.json', 'r') as json_file:
            self.index2char = json.load(json_file)
        with open('data/char2index.json', 'r') as json_file:
            self.char2index = json.load(json_file)

        self.avg_sentence_length = 165

        self.top_k = 3

        self.lm_unit(checkpoint_path, hyper_parameters)

    def lm_unit(self, checkpoint_path, hyper_parameters):
        self.rnn = CharRNN(len(self.index2char), hyper_parameters['num_layers'], hyper_parameters['hidden_size'])
        self.rnn.load_state_dict(torch.load(checkpoint_path, map_location=hyper_parameters['device']))

    def get_next_states_and_output(self, indexed_prefix, hidden_states):
        indexed_prefix = torch.tensor([indexed_prefix])
        indexed_prefix = torch.squeeze(indexed_prefix, dim=1).T
        output, hidden_states = self.rnn(indexed_prefix, hidden_states)
        return output, hidden_states

    def prefix_to_hiddens(self, indexed_prefix):
        _, hidden_states = self.get_next_states_and_output(indexed_prefix, None)
        return hidden_states

    def generate_new_sample(self, indexed_prefix):
        indexed_sentence = indexed_prefix
        last_generated_indexed_char = indexed_prefix[-1]

        while last_generated_indexed_char != self.char2index['</s>'] and \
                len(indexed_sentence) < self.avg_sentence_length:
            probs = self.get_probability(indexed_sentence)
            argsort_probs = np.argsort(probs.detach().numpy())
            random_choice_indexed_char = random.choice(argsort_probs[:self.top_k])
            if last_generated_indexed_char == self.char2index['N'] and \
                    random_choice_indexed_char == self.char2index['N']:
                continue
            last_generated_indexed_char = random_choice_indexed_char
            indexed_sentence.append(last_generated_indexed_char)

        return indexed_sentence

    def get_probability(self, indexed_prefix):
        output, _ = self.get_next_states_and_output(indexed_prefix, None)
        output = F.softmax(torch.squeeze(output), dim=0)
        return output[-1]

    def get_overall_probability(self, indexed_sentence):
        log_prob = 0
        for ind in range(2, len(indexed_sentence)):
            indexed_prefix = indexed_sentence[:ind]

            probs = self.get_probability(indexed_prefix)

            log_prob += math.log10(probs[indexed_sentence[ind]])

        return log_prob
