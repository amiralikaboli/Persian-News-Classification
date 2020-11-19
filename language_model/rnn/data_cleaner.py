import json
import re

import pandas as pd
import parsivar


class TextCleaner:
    def __init__(self):
        self.number_replacement = 'N'

        self.avg_word_length = 275

        self.punctuations = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@', '[',
            '\\', ']', '^', '_', '`', '{', '|', '}', '~', '£', '«', '®', '±', '»', '¼', '½', '×', 'ã', 'é', '÷', 'ø',
            'č', 'ı', 'š', 'ˈ', '˜', '˝', 'α', 'И', 'и', 'к', 'н', '٪', '٫', '٬', '–', '—', '’', '“', '”', '…', '‹',
            '›', '™', '♫', '❤', '《', '》', '', '﴾', '﴿', '：', '�'
        ]
        self.diacritics_pattern = re.compile(
            "[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670\u06d6\u06da]")
        self.emojis_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        self.english_characters_pattern = re.compile("[AАBCСDEFGHIJKКLMNOОPРQRSTUVWXYZaаbcсdefghijklmnopрqrstuvwxyуz]")
        self.numbers_pattern = re.compile("[0-9]+")
        self.space_patterns = [
            (re.compile("[\u202c\u2005\u2009\u2029\u2066\u3000\ufe0f]"), ' '),
            (re.compile("[\f\r\t\n]"), ' '),
            (re.compile("[\u200a\u200e\u200f\u206d\xa0\xad]"), '\u200c'),
            (re.compile("[\u200b\u200d\u202a\u202b\u2003\u2060\u2063\u2067\u2069\ufeff\x18]"), ''),
        ]

        self.normalizer = parsivar.Normalizer()

    def remove_punctuations(self, text: str) -> str:
        for punctuation in self.punctuations:
            text = text.replace(punctuation, ' ')
        return text

    def remove_diacritics(self, text: str) -> str:
        return self.diacritics_pattern.sub(r'', text)

    def remove_emojis(self, text: str) -> str:
        return self.emojis_pattern.sub(r'', text)

    def remove_english_characters(self, text: str) -> str:
        return self.english_characters_pattern.sub(r'', text)

    def mask_numbers(self, text: str) -> str:
        text = text.replace('٥', '5')
        text = self.numbers_pattern.sub(f' {self.number_replacement} ', text)
        return re.sub(f"{self.number_replacement}( *{self.number_replacement})*", self.number_replacement, text)

    def unify_spaces(self, text: str) -> str:
        for pattern, repl in self.space_patterns:
            text = pattern.sub(repl, text)
        return text

    def clean_text(self, text: str) -> str:
        text = self.remove_english_characters(text)
        text = self.remove_punctuations(text)
        text = self.remove_diacritics(text)
        text = self.remove_emojis(text)
        text = self.unify_spaces(text)
        text = self.normalizer.sub_alphabets(text)
        text = self.normalizer.space_correction(text)
        text = self.mask_numbers(text)

        text = text.replace('?', '؟')
        text = text.replace('  ', ' ')

        return text

    @staticmethod
    def tokenize(text: str) -> list:
        return list(text)

    def clean_training_set(self):
        training_data = []
        data = pd.read_csv('data/train.csv', encoding='utf-8', delimiter='\t')
        for ind in range(data.shape[0]):
            training_data.append(str(data['text'][ind]))

        training_data_tokens = []
        unique_chars = {'<s>', '</s>'}
        for training_text in training_data:
            if len(training_text.split()) > self.avg_word_length:
                continue

            cleaned_text = self.clean_text(training_text)
            tokens = ['<s>'] + self.tokenize(cleaned_text) + ['</s>']

            unique_chars = unique_chars.union(tokens)

            training_data_tokens.append(tokens)

        index2char = sorted(list(unique_chars))
        char2index = {ch: ind for ind, ch in enumerate(index2char)}

        for tokens in training_data_tokens:
            for ind, ch in enumerate(tokens):
                tokens[ind] = char2index[ch]

        with open('data/cleaned_train_tokens.json', 'w') as json_file:
            json.dump(training_data_tokens, json_file, ensure_ascii=False)
        with open('data/index2char.json', 'w') as json_file:
            json.dump(index2char, json_file, ensure_ascii=False)
        with open('data/char2index.json', 'w') as json_file:
            json.dump(char2index, json_file, ensure_ascii=False)

    def clean_test_set(self):
        test_data = []
        data = pd.read_csv('data/test.csv', encoding='utf-8', delimiter='\t')
        for ind in range(data.shape[0]):
            test_data.append(str(data['text'][ind]))

        with open('data/index2char.json', 'r', encoding='utf-8') as json_file:
            index2char = json.load(json_file)
        with open('data/char2index.json', 'r', encoding='utf-8') as json_file:
            char2index = json.load(json_file)

        test_data_tokens = []
        for test_text in test_data:
            if len(test_text.split()) > self.avg_word_length:
                continue

            cleaned_text = self.clean_text(test_text)
            tokens = ['<s>'] + self.tokenize(cleaned_text) + ['</s>']

            wordlevel_tokens = [char2index[ch] for ch in tokens]

            test_data_tokens.append(wordlevel_tokens)

        with open('data/cleaned_test_tokens.json', 'w') as json_file:
            json.dump(test_data_tokens, json_file, ensure_ascii=False)


if __name__ == '__main__':
    text_cleaner = TextCleaner()
    text_cleaner.clean_training_set()
    text_cleaner.clean_test_set()
