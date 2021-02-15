import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc

from text_cleaner import TextCleaner


class Classifier:
    def __init__(self, level: int):
        self.level = level

        self.text_cleaner = TextCleaner(level=level)

    def clean(self):
        self.text_cleaner.clean_training_set()
        self.text_cleaner.clean_test_set()

    def vectorize(self, cleaned_documents):
        result = {}

        num_documents = len(cleaned_documents)

        if self.level in [0, 2]:
            with open('data/index2word.json', 'r', encoding='utf-8') as json_file:
                index2word = json.load(json_file)
            with open('data/word2index.json', 'r', encoding='utf-8') as json_file:
                word2index = json.load(json_file)

            count_matrix = {}
            for ind, cleaned_document in enumerate(cleaned_documents):
                indexed_words = cleaned_document['words']

                for indexed_word in indexed_words:
                    if indexed_word not in count_matrix:
                        count_matrix[indexed_word] = {}
                    if ind not in count_matrix[indexed_word]:
                        count_matrix[indexed_word][ind] = 0
                    count_matrix[indexed_word][ind] += 1

            tfidf = np.zeros((num_documents, len(index2word)), dtype=np.uint8)
            for word_ind in range(len(index2word)):
                try:
                    for doc_ind in count_matrix[word_ind].keys():
                        tfidf[doc_ind][word_ind] = (1 + math.log10(count_matrix[word_ind][doc_ind])) * math.log10(
                            num_documents / len(count_matrix[word_ind]))
                except KeyError:
                    continue

            result['word-level'] = tfidf

        if self.level in [1, 2]:
            with open('data/index2char.json', 'r', encoding='utf-8') as json_file:
                index2char = json.load(json_file)
            with open('data/char2index.json', 'r', encoding='utf-8') as json_file:
                char2index = json.load(json_file)

            count_matrix = {}
            for ind, cleaned_document in enumerate(cleaned_documents):
                indexed_chars = cleaned_document['chars']

                for indexed_char in indexed_chars:
                    if indexed_char not in count_matrix:
                        count_matrix[indexed_char] = {}
                    if ind not in count_matrix[indexed_char]:
                        count_matrix[indexed_char][ind] = 0
                    count_matrix[indexed_char][ind] += 1

            tfidf = np.zeros((num_documents, len(index2char)), dtype=np.uint8)
            for char_ind in range(len(index2char)):
                try:
                    for doc_ind in count_matrix[char_ind].keys():
                        tfidf[doc_ind][char_ind] = (1 + math.log10(count_matrix[char_ind][doc_ind])) * math.log10(
                            num_documents / len(count_matrix[char_ind]))
                except KeyError:
                    continue

            result['char-level'] = tfidf

        return result

    def defining_model(self):
        if self.level in [0, 2]:
            self.best_word_measure = 0
            self.best_word_model = svm.SVC()

        if self.level in [1, 2]:
            self.best_char_measure = 0
            self.best_char_model = svm.SVC()

    def train(self):
        with open('data/training_cleaned_documents.json', 'r') as json_file:
            cleaned_documents = json.load(json_file)
        random.shuffle(cleaned_documents)
        cleaned_documents = cleaned_documents[:20000]

        categories = [cleaned_document['category'] for cleaned_document in cleaned_documents]

        num_chunks = 10
        chunk_length = int(math.ceil(len(cleaned_documents) / num_chunks))

        for validation_chunk_ind in range(num_chunks):
            word_model = svm.SVC()
            char_model = svm.SVC()

            start_index = validation_chunk_ind * chunk_length
            end_index = (validation_chunk_ind + 1) * chunk_length

            training_documents_chunks = cleaned_documents[:start_index] + cleaned_documents[end_index:]
            training_categories_chunks = categories[:start_index] + categories[end_index:]

            training_vectorize_result = self.vectorize(training_documents_chunks)

            if self.level in [0, 2]:
                tfidf_vectors_chunk = training_vectorize_result['word-level']

                word_model.fit(tfidf_vectors_chunk, training_categories_chunks)

            if self.level in [1, 2]:
                onehot_vectors_chunk = training_vectorize_result['char-level']

                char_model.fit(onehot_vectors_chunk, training_categories_chunks)

            validation_vectorize_result = self.vectorize(cleaned_documents[start_index:end_index])

            if self.level in [0, 2]:
                predictions = word_model.predict(validation_vectorize_result['word-level'])

                measure = sum(predictions == categories[start_index:end_index]) / len(predictions)

                if measure > self.best_word_measure:
                    self.best_word_model = word_model
                    self.best_word_measure = measure

            if self.level in [1, 2]:
                predictions = char_model.predict(validation_vectorize_result['char-level'])

                measure = sum(predictions == categories[start_index:end_index]) / len(predictions)

                if measure > self.best_char_measure:
                    self.best_char_model = char_model
                    self.best_char_measure = measure

    def evaluate(self):
        with open('data/test_cleaned_documents.json', 'r') as json_file:
            cleaned_documents = json.load(json_file)
        with open('data/category2index.json', 'r') as json_file:
            category2index = json.load(json_file)
        with open('data/index2category.json', 'r') as json_file:
            index2category = json.load(json_file)
        random.shuffle(cleaned_documents)
        # cleaned_documents = cleaned_documents[:2000]

        categories = [cleaned_document['category'] for cleaned_document in cleaned_documents]

        vectorize_result = self.vectorize(cleaned_documents)

        result = {}

        if self.level in [0, 2]:
            tfidf_vectors = vectorize_result['word-level']

            predictions = self.best_word_model.predict(tfidf_vectors)

            confusion_matrix = np.zeros((len(index2category), len(index2category)))
            for ind in range(len(categories)):
                actual_index = categories[ind]
                predicted_index = predictions[ind]

                confusion_matrix[actual_index][predicted_index] += 1

            accuracy = sum([confusion_matrix[ind][ind] for ind in range(len(index2category))]) / confusion_matrix.sum()
            precision = [
                confusion_matrix[ind][ind] / confusion_matrix.sum(axis=1)[ind]
                for ind in range(len(index2category))
            ]
            recall = [
                confusion_matrix[ind][ind] / confusion_matrix.sum(axis=0)[ind]
                for ind in range(len(index2category))
            ]
            f1_score = [
                2 * precision[ind] * recall[ind] / (precision[ind] + recall[ind])
                for ind in range(len(index2category))
            ]

            plt.figure(figsize=(32, 24))
            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            fpr = {}
            tpr = {}
            roc_auc = {}
            for category_val in np.unique(categories):
                y_true = categories == np.ones(len(categories)) * category_val
                y_score = predictions == np.ones(len(predictions)) * category_val

                fpr[category_val], tpr[category_val], _ = roc_curve(y_true, y_score)
                roc_auc[category_val] = auc(fpr[category_val], tpr[category_val])
                plt.plot(fpr[category_val], tpr[category_val], lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'.format(category_val, roc_auc[category_val]))

            plt.savefig('data/word-ROC.png')

            result['word-level'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': f1_score,
                'confusion-matrix': (confusion_matrix.T / confusion_matrix.sum(axis=1)).T.tolist(),
            }

        if self.level in [1, 2]:
            onehot_vectors = vectorize_result['char-level']

            predictions = self.best_char_model.predict(onehot_vectors)

            confusion_matrix = np.zeros((len(index2category), len(index2category)))
            for ind in range(len(categories)):
                actual_index = categories[ind]
                predicted_index = predictions[ind]

                confusion_matrix[actual_index][predicted_index] += 1

            accuracy = sum([confusion_matrix[ind][ind] for ind in range(len(index2category))]) / confusion_matrix.sum()
            precision = [
                confusion_matrix[ind][ind] / confusion_matrix.sum(axis=1)[ind]
                for ind in range(len(index2category))
            ]
            recall = [
                confusion_matrix[ind][ind] / confusion_matrix.sum(axis=0)[ind]
                for ind in range(len(index2category))
            ]
            f1_score = [
                2 * precision[ind] * recall[ind] / (precision[ind] + recall[ind])
                for ind in range(len(index2category))
            ]

            plt.figure(figsize=(32, 24))
            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            fpr = {}
            tpr = {}
            roc_auc = {}
            for category_val in np.unique(categories):
                y_true = categories == np.ones(len(categories)) * category_val
                y_score = predictions == np.ones(len(predictions)) * category_val

                fpr[category_val], tpr[category_val], _ = roc_curve(y_true, y_score)
                roc_auc[category_val] = auc(fpr[category_val], tpr[category_val])
                plt.plot(fpr[category_val], tpr[category_val], lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'.format(category_val, roc_auc[category_val]))

            plt.savefig('data/char-ROC.png')

            result['char-level'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': f1_score,
                'confusion-matrix': (confusion_matrix.T / confusion_matrix.sum(axis=1)).T.tolist(),
            }

        with open('data/evaluation_result.json', 'w') as json_file:
            json.dump(result, json_file)


if __name__ == '__main__':
    classifier = Classifier(level=0)
    classifier.defining_model()
    classifier.train()
    classifier.evaluate()
