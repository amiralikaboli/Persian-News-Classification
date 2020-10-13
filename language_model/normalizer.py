import json
import re

import hazm
import parsivar


def remove_punctuations(sentence):
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@',
                    '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '«', '»', '،', '؛', '٪', '٬', '…', '٫']
    for punctuation in punctuations:
        sentence = sentence.replace(punctuation, ' ')
    return sentence


def remove_diacritics(sentence):
    pattern = re.compile("[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655]")
    return pattern.sub(r'', sentence)


def remove_emojis(sentence):
    pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    return pattern.sub(r'', sentence)


def remove_english_characters(sentence):
    pattern = re.compile("[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]")
    return pattern.sub(r'', sentence)


def mask_numbers(sentence):
    pattern = re.compile("[-+]?[۰-۹]+(٫[۰-۹]+)?")
    return pattern.sub(' NUM ', sentence)


def normalizing_training_set():
    with open('data/train.json', 'r') as json_file:
        training_data = json.load(json_file)

    parsivar_normalizer = parsivar.Normalizer()
    hazm_normalizer = hazm.Normalizer()
    sentence_tokenizer = hazm.SentenceTokenizer()
    word_tokenizer = hazm.WordTokenizer(join_verb_parts=False)

    word_frequency = {}
    all_sentence_tokens = []
    for text in training_data:
        text = parsivar_normalizer.sub_alphabets(text)
        text = hazm_normalizer.normalize(text)
        text = remove_english_characters(text)
        text = mask_numbers(text)
        text = remove_punctuations(text)
        text = remove_diacritics(text)
        text = remove_emojis(text)

        text = text.replace('\n', ' ')
        text = text.replace('?', '؟')
        text = text.replace('؟', ' ؟ ')
        text = text.replace('.', ' . ')
        text = text.replace('  ', ' ')
        sentences = sentence_tokenizer.tokenize(text)

        for sentence in sentences:
            words = word_tokenizer.tokenize(sentence)

            if words[-1] == '.' or words[-1] == '؟':
                words = words[:-1]

            if len(words) == 0:
                continue

            for word in words:
                if word not in word_frequency:
                    word_frequency[word] = 0
                word_frequency[word] += 1

            all_sentence_tokens.append(words)

    with open('words_frequency.json', 'w') as json_file:
        json.dump(word_frequency, json_file, ensure_ascii=False)

    frequency_rank_threshold = 10000
    most_frequent_words = sorted(word_frequency, key=word_frequency.get, reverse=True)[:frequency_rank_threshold]

    final_all_sentence_tokens = []
    for sentence_tokens in all_sentence_tokens:
        final_sentence_tokens = []
        for ind, token in enumerate(sentence_tokens):
            if token == 'NUM':
                if len(final_sentence_tokens) == 0 or final_sentence_tokens[-1] != 'NUM':
                    final_sentence_tokens.append(token)
            elif token not in most_frequent_words:
                if len(final_sentence_tokens) == 0 or final_sentence_tokens[-1] != 'UNK':
                    final_sentence_tokens.append(token)
            else:
                final_sentence_tokens.append(token)
        final_all_sentence_tokens.append(final_sentence_tokens)

    with open('training_sentences.json', 'w') as json_file:
        json.dump(final_all_sentence_tokens, json_file, ensure_ascii=False)
    with open('most_frequent_words.json', 'w') as json_file:
        json.dump(most_frequent_words, json_file, ensure_ascii=False)


def normalizing_validation_set():
    with open('data/valid.json', 'r') as json_file:
        validation_data = json.load(json_file)

    with open('most_frequent_words.json', 'r') as json_file:
        most_frequent_words = json.load(json_file)

    parsivar_normalizer = parsivar.Normalizer()
    hazm_normalizer = hazm.Normalizer()
    sentence_tokenizer = hazm.SentenceTokenizer()
    word_tokenizer = hazm.WordTokenizer(join_verb_parts=False)

    all_sentence_tokens = []
    for text in validation_data:
        text = parsivar_normalizer.sub_alphabets(text)
        text = hazm_normalizer.normalize(text)
        text = remove_english_characters(text)
        text = mask_numbers(text)
        text = remove_punctuations(text)
        text = remove_diacritics(text)
        text = remove_emojis(text)

        text = text.replace('\n', ' ')
        text = text.replace('?', '؟')
        text = text.replace('؟', ' ؟ ')
        text = text.replace('.', ' . ')
        text = text.replace('  ', ' ')
        sentences = sentence_tokenizer.tokenize(text)

        for sentence in sentences:
            words = word_tokenizer.tokenize(sentence)

            if words[-1] == '.' or words[-1] == '؟':
                words = words[:-1]

            if len(words) == 0:
                continue
            final_sentence_tokens = []
            for ind, word in enumerate(words):
                if word == 'NUM':
                    if len(final_sentence_tokens) == 0 or final_sentence_tokens[-1] != 'NUM':
                        final_sentence_tokens.append(word)
                elif word not in most_frequent_words:
                    if len(final_sentence_tokens) == 0 or final_sentence_tokens[-1] != 'UNK':
                        final_sentence_tokens.append(word)
                else:
                    final_sentence_tokens.append(word)

            all_sentence_tokens.append(words)

    with open('validation_sentences.json', 'w') as json_file:
        json.dump(all_sentence_tokens, json_file, ensure_ascii=False)


if __name__ == '__main__':
    normalizing_training_set()
    normalizing_validation_set()
