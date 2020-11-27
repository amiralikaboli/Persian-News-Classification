import json

import torch

from language_model import LanguageModel


class LanguageModelEvaluator:
    def __init__(self, checkpoint_path, hyper_parameters):
        self.language_model = LanguageModel(checkpoint_path, hyper_parameters)

    def perplexity_log(self, indexed_sentence):
        overall_prob = self.language_model.get_overall_probability(indexed_sentence)

        return (-1 / len(indexed_sentence)) * overall_prob

    def char_error_rate(self, truth_indexed_sentence):
        generated_indexed_sentence = self.language_model.generate_new_sample(truth_indexed_sentence[:10])

        dp = [[0 for __ in range(len(generated_indexed_sentence) + 1)] for _ in range(len(truth_indexed_sentence) + 1)]

        for i in range(len(truth_indexed_sentence) + 1):
            for j in range(len(generated_indexed_sentence) + 1):
                if i == 0 or j == 0:
                    dp[0][j] = max(i, j)
                elif truth_indexed_sentence[i - 1] == generated_indexed_sentence[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

        return dp[len(truth_indexed_sentence)][len(generated_indexed_sentence)] / len(truth_indexed_sentence)


if __name__ == '__main__':
    with open('hyper_parameters.json', 'r') as json_file:
        hyper_parameters = json.load(json_file)
    hyper_parameters['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_evaluator = LanguageModelEvaluator('data/checkpoints/2020-11-25 23_25_55.536518.pth', hyper_parameters)
