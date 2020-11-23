import json
import re

import pandas as pd
import parsivar


class TextCleaner:
    def __init__(self):
        self.number_replacement = 'N'

        self.avg_sentence_length = 165

        self.punctuations = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@', '[',
            '\\', ']', '^', '_', '`', '{', '|', '}', '~', '£', '¤', '§', '©', '«', '®', '°', '±', '²', '´', '¸', '»',
            '¼', '½', '¾', '×', '÷', 'ˈ', '˜', '˝', '٪', '٫', '٬', '‐', '–', '—', '‘', '’', '“', '”', '„', '…', '″',
            '‹', '›', '™', '↑', '→', '↓', '⋅', '⌘', '▪', '◄', '○', '♫', '✓', '❤', '《', '》', '爆', '者', '被', '\uf020',
            '\uf04f', '\uf05f', '\uf076', '\uf0a7', '\uf0fc', '﴾', '﴿', '：', '�'
        ]
        self.diacritics_pattern = re.compile("[\u064B-\u065e\u0670\u0674\u06c3\u06d4-\u06ed]")
        self.emojis_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        self.latin_characters_pattern = re.compile(
            "["
            "\u0041-\u007a"
            "\u00c0-\u036f"
            "\u0400-\u050f"
            "\u0342-\u03ff"
            "]"
        )
        self.numbers_pattern = re.compile("[0-9]+")
        self.space_patterns = [
            (re.compile("[\u202c\u2005\u2009\u2029\u2066\u3000\ufe0f]"), ' '),
            (re.compile("[\f\r\t\n]"), ' '),
            (re.compile("[\u001f\u009d\u200a\u200e\u200f\u206d\xa0\xad]"), '\u200c'),
            (re.compile("[\u007f\u0085\u061c\u200b\u200d\u202a\u202b\u206f\u2003"
                        "\u2028\u2060\u2063\u2067\u2069\ufeff\ufffc\x18]"), ''),
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

    def remove_latin_characters(self, text: str) -> str:
        return self.latin_characters_pattern.sub(r'', text)

    def mask_numbers(self, text: str) -> str:
        text = text.replace('٥', '5')
        text = self.numbers_pattern.sub(f' {self.number_replacement} ', text)
        return re.sub(f"{self.number_replacement}( *{self.number_replacement})*", self.number_replacement, text)

    def unify_spaces(self, text: str) -> str:
        for pattern, repl in self.space_patterns:
            text = pattern.sub(repl, text)
        return text

    @staticmethod
    def tokenize(text):
        return list(text) if text[0] != ' ' else list(text[1:])

    def clean_text(self, text: str) -> str:
        text = self.remove_latin_characters(text)
        text = self.remove_punctuations(text)
        text = self.remove_diacritics(text)
        text = self.remove_emojis(text)
        text = self.unify_spaces(text)
        text = self.normalizer.sub_alphabets(text)
        text = self.normalizer.space_correction(text)
        text = self.mask_numbers(text)

        text = text.replace('؛', '،')
        text = text.replace('?', '؟')
        text = text.replace('  ', ' ')

        return text

    @staticmethod
    def get_sentences(text):
        separator_indexes = [match.start() for match in re.finditer("[.؟]", text)]
        start_sentence_indexes = [0] + [separator_index + 1 for separator_index in separator_indexes] + [len(text)]

        sentences = [
            text[start_sentence_indexes[ind]:start_sentence_indexes[ind + 1]]
            for ind in range(len(start_sentence_indexes) - 1)
        ]

        return sentences

    def clean_training_set(self):
        training_data = []
        data = pd.read_csv('data/train.csv', encoding='utf-8', delimiter='\t')
        for ind in range(data.shape[0]):
            training_data.append(str(data['text'][ind]))

        training_sentences = []
        unique_chars = {'<s>', '</s>'}
        for training_text in training_data:
            cleaned_text = self.clean_text(training_text)
            sentences = self.get_sentences(cleaned_text)

            for sentence in sentences:
                if len(sentence) < 5 or len(sentence) > self.avg_sentence_length:
                    continue

                sentence_chars = ['<s>', *self.tokenize(sentence), '</s>']

                unique_chars = unique_chars.union(sentence_chars)

                training_sentences.append(sentence_chars)

        index2char = sorted(list(unique_chars))
        char2index = {ch: ind for ind, ch in enumerate(index2char)}

        for ind, training_sentence in enumerate(training_sentences):
            training_sentences[ind] = [char2index[ch] for ch in training_sentence]

        with open('data/train_sentences.json', 'w') as json_file:
            json.dump(training_sentences, json_file, ensure_ascii=False)
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

        test_sentences = []
        for test_text in test_data:
            cleaned_text = self.clean_text(test_text)
            sentences = self.get_sentences(cleaned_text)

            for sentence in sentences:
                if len(sentence) < 5 or len(sentence) > self.avg_sentence_length:
                    continue

                sentence_chars = ['<s>', *self.tokenize(sentence), '</s>']

                indexed_sentence_chars = [char2index[ch] for ch in sentence_chars if ch in char2index]

                test_sentences.append(indexed_sentence_chars)

        with open('data/test_sentences.json', 'w') as json_file:
            json.dump(test_sentences, json_file, ensure_ascii=False)


if __name__ == '__main__':
    text_cleaner = TextCleaner()
    text_cleaner.clean_training_set()
    text_cleaner.clean_test_set()
