import json

import torch

from language_model import LanguageModel


class LanguageModelEvaluator:
    def __init__(self, checkpoint_path, hyper_parameters):
        self.language_model = LanguageModel(checkpoint_path, hyper_parameters)

    def perplexity_log(self, indexed_sentence):
        overall_prob = self.language_model.get_overall_probability(indexed_sentence)

        return (-1 / len(indexed_sentence)) * overall_prob, overall_prob

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

        return dp[len(truth_indexed_sentence)][len(generated_indexed_sentence)] / len(
            truth_indexed_sentence), generated_indexed_sentence

    def evaluate_test_set(self):
        with open('data/test_sentences.json', 'r', encoding='utf-8') as json_file:
            test_sentences = json.load(json_file)

        sum_perplexity = 0
        sum_cer = 0
        generated_sentences = []
        probabilities_log = []
        for indexed_sentence in test_sentences:
            perplexity, overall_prob = self.perplexity_log(indexed_sentence)
            cer, generated_indexed_sentence = self.char_error_rate(indexed_sentence)

            sum_perplexity += perplexity
            sum_cer += cer

            generated_sentences.append(
                ' '.join([self.language_model.index2char[indexed_char] for indexed_char in generated_indexed_sentence])
            )
            probabilities_log.append(overall_prob)

        with open('data/new_samples.json', 'w') as json_file:
            json.dump(generated_sentences, json_file, ensure_ascii=False)
        with open('data/probabilities.json', 'w') as json_file:
            json.dump(probabilities_log, json_file, ensure_ascii=False)

        return sum_perplexity / len(test_sentences), sum_cer / len(test_sentences)


if __name__ == '__main__':
    with open('hyper_parameters.json', 'r') as json_file:
        hyper_parameters = json.load(json_file)
    hyper_parameters['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_evaluator = LanguageModelEvaluator('data/checkpoints/2020-11-28 20_16_27.077470.pth', hyper_parameters)

    avg_perplexity, avg_cer = lm_evaluator.evaluate_test_set()

    print(avg_perplexity, avg_cer)
