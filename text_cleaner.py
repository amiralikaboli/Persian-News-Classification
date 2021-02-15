import json
import re

import hazm
import pandas as pd
import parsivar


class TextCleaner:
    def __init__(self, level: int):
        self.level = level

        self.number_replacement = 'N'
        self.unknown_replacement = 'UNK'
        self.pad_replacement = 'PAD'

        self.avg_word_length = 180
        self.avg_char_length = 1500

        self.punctuations = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@', '[',
            '\\', ']', '^', '_', '`', '{', '|', '}', '~', '«', '»', '،', '؛', '٪', '٬', '…', '٫', '：', '—', '‹', '›'
        ]
        self.diacritics_pattern = re.compile("[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655]")
        self.emojis_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        self.english_characters_pattern = re.compile("[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]")
        self.numbers_pattern = re.compile("[-+]?[۰-۹]+(٫[۰-۹]+)?")
        self.half_space_pattern = re.compile("[\u200C\u200f\xa0]")
        self.stopwords = hazm.stopwords_list()[:200] + [
            'ام', 'م', 'ات', 'ای', 'ی', 'ت', 'اش', 'ش', 'مان', 'یم', 'ایم', 'تان', 'ید', 'اید', 'شان', 'ند', 'اند',
            'است', 'هست', 'بود', 'شد', 'شو', 'باش', 'خواه', 'ها', 'های', 'ان', 'هستم', 'هستم', 'هست', 'هستید', 'هستیم',
            'نیستم', 'نیستی', 'نیست', 'نیستیم', 'نیستید', 'نیستند'
        ]

        self.stemmer = parsivar.FindStems()  # # delimiter
        self.lemmatizer = hazm.Lemmatizer()  # & delimiter
        self.parsivar_normalizer = parsivar.Normalizer()
        self.hazm_normalizer = hazm.Normalizer()
        self.tokenizer = hazm.WordTokenizer(join_verb_parts=False)

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

    def remove_half_space(self, text: str) -> str:
        return self.half_space_pattern.sub(r' ', text)

    def mask_numbers(self, text: str) -> str:
        return self.numbers_pattern.sub(f' {self.number_replacement} ', text)

    def stem(self, token: str) -> str:
        stemmed_word = self.stemmer.convert_to_stem(self.lemmatizer.lemmatize(token)).replace('&', '#')
        if '#' in stemmed_word:
            past, present = stemmed_word.split('#')
            stemmed_word = past if past in token else present
        return stemmed_word

    def clean_text(self, text):
        text = self.parsivar_normalizer.sub_alphabets(text)
        text = self.hazm_normalizer.normalize(text)
        text = self.remove_english_characters(text)
        text = self.mask_numbers(text)
        text = self.remove_punctuations(text)
        text = self.remove_diacritics(text)
        text = self.remove_emojis(text)
        text = self.remove_half_space(text)

        text = text.replace('\n', ' ')
        text = text.replace('?', '؟')
        text = text.replace('؟', ' ؟ ')
        text = text.replace('.', ' . ')
        text = text.replace('  ', ' ')

        return text

    def tokenize(self, text: str) -> dict:
        result = {}

        if self.level in [0, 2]:
            tokens = self.tokenizer.tokenize(text)

            stemmed_words = []
            for token in tokens:
                stemmed_word = self.stem(token)
                if token not in self.stopwords and stemmed_word not in self.stopwords:
                    stemmed_words.append(stemmed_word)

            result['word-level'] = stemmed_words

        if self.level in [1, 2]:
            result['char-level'] = list(text)

        return result

    def clean_training_set(self):
        training_data = []
        data = pd.read_csv('data/train.csv', encoding='utf-8', delimiter='\t')
        for ind in range(data.shape[0]):
            training_data.append({
                'title': str(data['title'][ind]),
                'text': str(data['text'][ind]),
                'category': str(data['category'][ind])
            })

        word_frequency = {}
        chars_frequency = {}
        index2category = []
        category2index = {}
        cleaned_documents = []
        for data_point in training_data:
            content = data_point['title'] + data_point['text']
            if data_point['category'] not in category2index:
                category2index[data_point['category']] = len(index2category)
                index2category.append(data_point['category'])
            category = category2index[data_point['category']]

            cleaned_text = self.clean_text(content)
            tokenize_result = self.tokenize(cleaned_text)

            if 'word-level' in tokenize_result and len(tokenize_result['word-level']) > self.avg_word_length:
                continue
            if 'char-level' in tokenize_result and len(tokenize_result['char-level']) > self.avg_char_length:
                continue

            cleaned_documents.append({'category': category})
            if self.level in [0, 2]:
                cleaned_tokens = tokenize_result['word-level']
                cleaned_documents[-1]['words'] = cleaned_tokens

                for word in cleaned_tokens:
                    if word not in word_frequency:
                        word_frequency[word] = 0
                    word_frequency[word] += 1

            if self.level in [1, 2]:
                cleaned_tokens = tokenize_result['char-level']
                cleaned_documents[-1]['chars'] = cleaned_tokens

                for ch in cleaned_tokens:
                    if ch not in chars_frequency:
                        chars_frequency[ch] = 0
                    chars_frequency[ch] += 1

        with open('data/category2index.json', 'w') as json_file:
            json.dump(category2index, json_file, ensure_ascii=False)
        with open('data/index2category.json', 'w') as json_file:
            json.dump(index2category, json_file, ensure_ascii=False)

        if self.level in [0, 2]:
            with open('data/words_frequency.json', 'w') as json_file:
                json.dump(word_frequency, json_file, ensure_ascii=False)

            rank_threshold = 10000
            most_frequent_words = sorted(word_frequency, key=word_frequency.get, reverse=True)[:rank_threshold]

            index2word = sorted(most_frequent_words)
            word2index = {word: ind for ind, word in enumerate(index2word)}
            word2index[self.unknown_replacement] = len(index2word)
            index2word.append(self.unknown_replacement)
            word2index[self.pad_replacement] = len(index2word)
            index2word.append(self.pad_replacement)

            with open('data/index2word.json', 'w') as json_file:
                json.dump(index2word, json_file, ensure_ascii=False)
            with open('data/word2index.json', 'w') as json_file:
                json.dump(word2index, json_file, ensure_ascii=False)
            with open('data/most_frequent_words.json', 'w') as json_file:
                json.dump(most_frequent_words, json_file, ensure_ascii=False)

            for ind, cleaned_document in enumerate(cleaned_documents):
                words = cleaned_document['words']

                indexed_document_words = []
                for word in words:
                    if word == self.number_replacement:
                        if len(indexed_document_words) == 0 or indexed_document_words[-1] != self.number_replacement:
                            indexed_document_words.append(word2index[self.number_replacement])
                    elif word not in most_frequent_words:
                        if len(indexed_document_words) == 0 or indexed_document_words[-1] != self.unknown_replacement:
                            indexed_document_words.append(word2index[self.unknown_replacement])
                    else:
                        indexed_document_words.append(word2index[word])

                cleaned_documents[ind]['words'] = indexed_document_words + [word2index[self.pad_replacement]] * (
                        self.avg_word_length - len(indexed_document_words))

        if self.level in [1, 2]:
            with open('data/chars_frequency.json', 'w') as json_file:
                json.dump(chars_frequency, json_file, ensure_ascii=False)

            index2char = sorted(list(chars_frequency.keys()))
            char2index = {ch: ind for ind, ch in enumerate(index2char)}
            char2index[self.unknown_replacement] = len(index2char)
            index2char.append(self.unknown_replacement)
            char2index[self.pad_replacement] = len(index2char)
            index2char.append(self.pad_replacement)

            with open('data/index2char.json', 'w') as json_file:
                json.dump(index2char, json_file, ensure_ascii=False)
            with open('data/char2index.json', 'w') as json_file:
                json.dump(char2index, json_file, ensure_ascii=False)

            for ind, cleaned_document in enumerate(cleaned_documents):
                chars = cleaned_document['chars']

                indexed_document_chars = []
                for ch in chars:
                    if ch == self.number_replacement:
                        if len(indexed_document_chars) == 0 or indexed_document_chars[-1] != self.number_replacement:
                            indexed_document_chars.append(char2index[self.number_replacement])
                    else:
                        indexed_document_chars.append(char2index[ch])

                cleaned_documents[ind]['chars'] = indexed_document_chars + [char2index[self.pad_replacement]] * (
                        self.avg_char_length - len(indexed_document_chars))

        with open('data/training_cleaned_documents.json', 'w') as json_file:
            json.dump(cleaned_documents, json_file, ensure_ascii=False)

    def clean_test_set(self):
        test_data = []
        data = pd.read_csv('data/test.csv', encoding='utf-8', delimiter='\t')
        for ind in range(data.shape[0]):
            test_data.append({
                'title': str(data['title'][ind]),
                'text': str(data['text'][ind]),
                'category': str(data['category'][ind])
            })

        if self.level in [0, 2]:
            with open('data/most_frequent_words.json', 'r', encoding='utf-8') as json_file:
                most_frequent_words = json.load(json_file)
            with open('data/index2word.json', 'r', encoding='utf-8') as json_file:
                index2word = json.load(json_file)
            with open('data/word2index.json', 'r', encoding='utf-8') as json_file:
                word2index = json.load(json_file)

        if self.level in [1, 2]:
            with open('data/index2char.json', 'r', encoding='utf-8') as json_file:
                index2char = json.load(json_file)
            with open('data/char2index.json', 'r', encoding='utf-8') as json_file:
                char2index = json.load(json_file)

        with open('data/category2index.json', 'r') as json_file:
            category2index = json.load(json_file)
        with open('data/index2category.json', 'r') as json_file:
            index2category = json.load(json_file)

        cleaned_documents = []
        for data_point in test_data:
            content = data_point['title'] + data_point['text']
            category = category2index[data_point['category']]

            cleaned_text = self.clean_text(content)
            tokenize_result = self.tokenize(cleaned_text)

            cleaned_documents.append({'category': category})
            if self.level in [0, 2]:
                cleaned_tokens = tokenize_result['word-level']

                document_words = []
                for ind, word in enumerate(cleaned_tokens):
                    if word == self.number_replacement:
                        if len(document_words) == 0 or document_words[-1] != self.number_replacement:
                            document_words.append(word2index[self.number_replacement])
                    elif word not in most_frequent_words:
                        if len(document_words) == 0 or document_words[-1] != self.unknown_replacement:
                            document_words.append(word2index[self.unknown_replacement])
                    else:
                        document_words.append(word2index[word])

                cleaned_documents[-1]['words'] = document_words + [word2index[self.pad_replacement]] * (
                        self.avg_word_length - len(document_words))

            if self.level in [1, 2]:
                cleaned_tokens = tokenize_result['char-level']

                document_chars = []
                for ind, ch in enumerate(cleaned_tokens):
                    if ch == self.number_replacement:
                        if len(document_chars) == 0 or document_chars[-1] != self.number_replacement:
                            document_chars.append(char2index[self.number_replacement])
                    if ch not in char2index:
                        document_chars.append(char2index[self.unknown_replacement])
                    else:
                        document_chars.append(char2index[ch])

                cleaned_documents[-1]['chars'] = document_chars + [char2index[self.pad_replacement]] * (
                        self.avg_char_length - len(document_chars))

        with open('data/test_cleaned_documents.json', 'w') as json_file:
            json.dump(cleaned_documents, json_file, ensure_ascii=False)


if __name__ == '__main__':
    text_cleaner = TextCleaner(level=2)  # 0: w, 1: c, 2: wc
    text_cleaner.clean_training_set()
    text_cleaner.clean_test_set()
