import json
import math
from typing import List, Tuple

from jiwer import wer


def logarithm(num):
    return math.log10(num) if num else -math.inf


class LanguageModel:
    def __init__(self, corpus_path: str, words_file_path: str, n_gram: int, smoothing_type: str = None):
        self.corpus_path = corpus_path

        if n_gram not in [1, 2, 3]:
            raise Exception('n-gram must between 1 to 3!')
        self.n_gram = n_gram

        if smoothing_type not in [None, 'laplace', 'kneser-ney']:
            raise Exception('invalid smoothing type!')
        self.smoothing_type = smoothing_type

        if n_gram == 1 and smoothing_type == 'kneser-ney':
            raise Exception('for kneser-ney smoothing, n-gram must greater than 1!')

        with open(words_file_path, 'r', encoding='utf-8') as json_file:
            self.word_list = json.load(json_file)
        self.word_list.append('UNK')

    def train(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as json_file:
            training_data = json.load(json_file)

        if self.n_gram == 1:
            self.unigram_frequency = {}
            self.most_frequent_unigram = {'token': '', 'freq': 0}
            for sentence in training_data:
                for token in sentence:
                    if token not in self.unigram_frequency:
                        self.unigram_frequency[token] = 0
                    self.unigram_frequency[token] += 1

                    if self.most_frequent_unigram['freq'] < self.unigram_frequency[token]:
                        self.most_frequent_unigram = {'token': token, 'freq': self.unigram_frequency[token]}

            with open('data/unigram_frequency.json', 'w') as json_file:
                json.dump(self.unigram_frequency, json_file, ensure_ascii=False)

        if self.n_gram == 2:
            self.bigram_frequency = {}
            self.most_frequent_bigrams = {}
            for sentence in training_data:
                bigram_tokens = ['<s>'] + sentence + ['</s>']
                for ind, token in enumerate(bigram_tokens[:-1]):
                    word1 = bigram_tokens[ind]
                    word2 = bigram_tokens[ind + 1]

                    if word1 not in self.bigram_frequency:
                        self.bigram_frequency[word1] = {}
                        self.most_frequent_bigrams[word1] = {'token': '', 'freq': 0}
                    if word2 not in self.bigram_frequency[word1]:
                        self.bigram_frequency[word1][word2] = 0
                    self.bigram_frequency[word1][word2] += 1

                    if self.most_frequent_bigrams[word1]['freq'] < self.bigram_frequency[word1][word2]:
                        self.most_frequent_bigrams[word1] = {
                            'token': word2,
                            'freq': self.bigram_frequency[word1][word2]
                        }

            with open('data/bigram_frequency.json', 'w') as json_file:
                json.dump(self.bigram_frequency, json_file, ensure_ascii=False)

            if self.smoothing_type == 'kneser-ney':
                self.unigrams_before_words = {}
                for sentence in training_data:
                    bigram_tokens = ['<s>'] + sentence + ['</s>']
                    for ind, token in enumerate(bigram_tokens[:-1]):
                        word1 = bigram_tokens[ind]
                        word2 = bigram_tokens[ind + 1]

                        if word2 not in self.unigrams_before_words:
                            self.unigrams_before_words[word2] = set()
                        self.unigrams_before_words[word2].add(word1)

                with open('data/unigrams_before_words.json', 'w') as json_file:
                    json.dump(list(self.unigrams_before_words), json_file, ensure_ascii=False)

        if self.n_gram == 3:
            self.trigram_frequency = {}
            self.most_frequent_trigrams = {}
            for sentence in training_data:
                trigram_tokens = ['<s>', '<s>'] + sentence + ['</s>', '</s>']
                for ind, token in enumerate(trigram_tokens[:-2]):
                    word1 = trigram_tokens[ind]
                    word2 = trigram_tokens[ind + 1]
                    word3 = trigram_tokens[ind + 2]

                    if word1 not in self.trigram_frequency:
                        self.trigram_frequency[word1] = {}
                        self.most_frequent_trigrams[word1] = {}
                    if word2 not in self.trigram_frequency[word1]:
                        self.trigram_frequency[word1][word2] = {}
                        self.most_frequent_trigrams[word1][word2] = {'token': '', 'freq': 0}
                    if word3 not in self.trigram_frequency[word1][word2]:
                        self.trigram_frequency[word1][word2][word3] = 0
                    self.trigram_frequency[word1][word2][word3] += 1

                    if self.most_frequent_trigrams[word1][word2]['freq'] < self.trigram_frequency[word1][word2][word3]:
                        self.most_frequent_trigrams[word1][word2] = {
                            'token': word3,
                            'freq': self.trigram_frequency[word1][word2][word3]
                        }

            with open('data/trigram_frequency.json', 'w') as json_file:
                json.dump(self.trigram_frequency, json_file, ensure_ascii=False)

            if self.smoothing_type == 'kneser-ney':
                self.bigrams_before_words = {}
                for sentence in training_data:
                    trigram_tokens = ['<s>', '<s>'] + sentence + ['</s>', '</s>']
                    for ind, token in enumerate(trigram_tokens[:-2]):
                        word1 = trigram_tokens[ind]
                        word2 = trigram_tokens[ind + 1]
                        word3 = trigram_tokens[ind + 2]

                        if word3 not in self.bigrams_before_words:
                            self.bigrams_before_words[word3] = set()
                        self.bigrams_before_words[word3].add((word1, word2))

                with open('data/bigrams_before_words.json', 'w') as json_file:
                    json.dump(list(self.bigrams_before_words), json_file, ensure_ascii=False)

    def smoothing(self, ngram_words: Tuple) -> float:
        if self.n_gram == 1:
            if ngram_words[0] not in self.unigram_frequency:
                freq = 0
            else:
                freq = self.unigram_frequency[ngram_words[0]]
            dict_freqs = self.unigram_frequency

            if self.smoothing_type is None:
                try:
                    return freq / sum(dict_freqs.values())
                except ZeroDivisionError:
                    return 0
            if self.smoothing_type == 'laplace':
                return (freq + 1) / (sum(dict_freqs.values()) + len(self.word_list))

        if self.n_gram == 2:
            if ngram_words[0] not in self.bigram_frequency:
                freq = 0
                dict_freqs = {}
            elif ngram_words[1] not in self.bigram_frequency[ngram_words[0]]:
                freq = 0
                dict_freqs = self.bigram_frequency[ngram_words[0]]
            else:
                freq = self.bigram_frequency[ngram_words[0]][ngram_words[1]]
                dict_freqs = self.bigram_frequency[ngram_words[0]]

            if self.smoothing_type is None:
                try:
                    return freq / sum(dict_freqs.values())
                except ZeroDivisionError:
                    return 0
            if self.smoothing_type == 'laplace':
                return (freq + 1) / (sum(dict_freqs.values()) + len(self.word_list))
            if self.smoothing_type == 'kneser-ney':
                delta = 0.5
                alpha = 0.5
                if freq:
                    return (freq - delta) / sum(dict_freqs.values())
                return alpha * len(self.unigrams_before_words[ngram_words[1]]) / sum(
                    [len(self.unigrams_before_words[word]) for word in self.unigrams_before_words])

        if self.n_gram == 3:
            if ngram_words[0] not in self.trigram_frequency or \
                    ngram_words[1] not in self.trigram_frequency[ngram_words[0]]:
                freq = 0
                dict_freqs = {}
            elif ngram_words[2] not in self.trigram_frequency[ngram_words[0]][ngram_words[1]]:
                freq = 0
                dict_freqs = self.trigram_frequency[ngram_words[0]][ngram_words[1]]
            else:
                freq = self.trigram_frequency[ngram_words[0]][ngram_words[1]][ngram_words[2]]
                dict_freqs = self.trigram_frequency[ngram_words[0]][ngram_words[1]]

            if self.smoothing_type is None:
                try:
                    return freq / sum(dict_freqs.values())
                except ZeroDivisionError:
                    return 0
            if self.smoothing_type == 'laplace':
                return (freq + 1) / (sum(dict_freqs.values()) + len(self.word_list))
            if self.smoothing_type == 'kneser-ney':
                delta = 0.5
                alpha = 0.5
                if freq:
                    return (freq - delta) / sum(dict_freqs.values())
                return alpha * len(self.bigrams_before_words[ngram_words[2]]) / sum(
                    [len(self.bigrams_before_words[word]) for word in self.bigrams_before_words])

    def prob_log(self, tokens: List[str]) -> float:
        overall_prob = 0

        if self.n_gram == 1:
            for word in tokens:
                unigram_prob = self.smoothing((word,))
                overall_prob += logarithm(unigram_prob)

        if self.n_gram == 2:
            for ind in range(len(tokens) - 1):
                word1 = tokens[ind]
                word2 = tokens[ind + 1]

                bigram_prob = self.smoothing((word1, word2))
                overall_prob += logarithm(bigram_prob)

        if self.n_gram == 3:
            for ind in range(len(tokens) - 2):
                word1 = tokens[ind]
                word2 = tokens[ind + 1]
                word3 = tokens[ind + 2]

                trigram_prob = self.smoothing((word1, word2, word3))
                overall_prob += logarithm(trigram_prob)

        return overall_prob

    def generate(self, tokens: List[str]) -> str:
        if self.n_gram == 1:
            return self.most_frequent_unigram['token']

        if self.n_gram == 2:
            return self.most_frequent_bigrams[tokens[-1]]['token']

        if self.n_gram == 3:
            return self.most_frequent_trigrams[tokens[-2]][tokens[-1]]['token']

    def evaluate(self, validation_sentences_path: str) -> float:
        with open(validation_sentences_path, 'r', encoding='utf-8') as json_file:
            validation_sentences = json.load(json_file)

        word_error_rates = []
        for validation_sentence in validation_sentences:
            validation_tokens = ['<s>'] + validation_sentence + ['</s>']

            generated_words = ['<s>']
            try:
                generated_words.append(validation_sentence[0])

                generated_words.append(self.generate(generated_words))
                while generated_words[-1] != '</s>' and len(generated_words) < 35:
                    generated_words.append(self.generate(generated_words))

                word_error_rates.append(wer(' '.join(validation_tokens), ' '.join(generated_words)))
            except KeyError:
                generated_words.append('<s>')

                generated_words.append(self.generate(generated_words))
                while generated_words[-1] != '</s>' and len(generated_words) < 35:
                    generated_words.append(self.generate(generated_words))

                word_error_rates.append(wer(' '.join(validation_tokens), ' '.join(generated_words[1:])))

        with open('data/word_error_rates.json', 'w') as json_file:
            json.dump(word_error_rates, json_file)

        return sum(word_error_rates) / len(word_error_rates)


if __name__ == '__main__':
    language_model = LanguageModel(
        corpus_path='data/training_sentences.json',
        words_file_path='data/most_frequent_words.json',
        n_gram=2,
        smoothing_type='laplace'
    )
    language_model.train()

    # avg_wer = language_model.evaluate('data/validation_sentences.json')
    # print(avg_wer)
