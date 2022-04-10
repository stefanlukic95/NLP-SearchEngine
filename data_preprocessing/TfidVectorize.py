from typing import Sequence
from numpy.linalg import norm
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np


# utility function for nomalization between [-1,1]
def normalize_negative_one(data):
    normalized_input = (data - np.min(data)) / (np.max(data) - np.min(data))
    return 2 * normalized_input - 1


class TfidVectorize:

    def __init__(self, questions: Sequence[str]):
        self.questions = questions

    # Output: word, dimension number pair for each word as a python dictionary
    def fit_custom(self, questions: Sequence[str]):
        unique_words = []
        for row in questions:
            for word in row.split(" "):
                if len(word) >= 2 and word not in unique_words:
                    unique_words.append(word)
        unique_words.sort()
        word_dimension_dict = {j: i for i, j in enumerate(
            unique_words)}
        return word_dimension_dict

    # Utility function number of words in dataset
    def count_of_words_in_whole_dataset(self, questions: Sequence[str], word):
        count = 0
        for row in questions:
            if word in row:
                count = count + 1
        return count

    def transform_custom(self, questions: Sequence[str], word_dimension_dict) -> np.ndarray:
        rows = []
        columns = []
        values = []
        tf_val = []
        idf_val = []

        for idx, row in enumerate(questions):
            word_freq = dict(Counter(row.split()))

            for word, freq in word_freq.items():
                if len(word) < 2:
                    continue

                col_index = word_dimension_dict.get(word, -1)
                if col_index != -1:
                    rows.append(idx)
                    columns.append(col_index)

                    tf_idf_value = (freq / len(row.split())) * (1 + (
                        np.log((int(len(questions))) / (self.count_of_words_in_whole_dataset(questions, word)))))

                    values.append(tf_idf_value)
                    sparse_matrix = csr_matrix((values, (rows, columns)),
                                               shape=(len(questions), len(word_dimension_dict)))

                    final_normalized_output = (sparse_matrix - np.min(sparse_matrix)) / (
                            np.max(sparse_matrix) - np.min(sparse_matrix))

        return final_normalized_output

    def cosine_similarity(self, query_vector: np.ndarray, corpus_vectors: np.ndarray):
        cosine_similarity_res = np.array([])

        for i in range(corpus_vectors.shape[0]):
            cos_similarity = np.dot(np.squeeze(query_vector.toarray()), np.squeeze(corpus_vectors[i].toarray())) \
                             / norm(np.squeeze(query_vector.toarray())) \
                             * norm(np.squeeze(corpus_vectors[i].toarray()))

            cosine_similarity_res = np.append(cosine_similarity_res, cos_similarity)
            np.seterr(divide='ignore', invalid='ignore')
            cos_similarity_normalized = normalize_negative_one(cosine_similarity_res)

        sorted_result_array = np.argsort(cos_similarity_normalized)
        sorted_array = cos_similarity_normalized[sorted_result_array]
        top_results = sorted_array[-5:]

        top_indices = (-cos_similarity_normalized).argsort()[:5]

        return cos_similarity_normalized, top_results[::-1], top_indices
