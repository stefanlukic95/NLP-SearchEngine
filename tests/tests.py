from typing import Sequence
import numpy as np
from numpy.linalg import norm


# functions
def fit_custom(questions: Sequence[str]):
    unique_words = []
    for row in questions:
        for word in row.split(" "):
            if len(word) >= 2 and word not in unique_words:
                unique_words.append(word)
    unique_words.sort()
    word_dimension_dict = {j: i for i, j in enumerate(
        unique_words)}
    return word_dimension_dict


def cosine_similarity(query_vector: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    cosine_similarity_res = np.array([])

    for i in range(corpus_vectors.shape[0]):
        cos_similarity = np.dot(np.squeeze(query_vector), np.squeeze(corpus_vectors[i])) \
                         / norm(np.squeeze(query_vector)) \
                         * norm(np.squeeze(corpus_vectors[i]))
        cosine_similarity_res = np.append(cosine_similarity_res, cos_similarity)
    return cosine_similarity_res


def normalize_negative_one(data):
    normalized_input = (data - np.min(data)) / (np.max(data) - np.min(data))
    return 2 * normalized_input - 1


# TESTS
def test_cosine_similarity():
    query_vector = np.array([2])
    corpus_vectors = np.array([3])
    cosine_result = cosine_similarity(query_vector, corpus_vectors)
    expected = 9
    assert cosine_result == expected


def test_fit_custom():
    questions = ["is java good"]
    result = fit_custom(questions)
    assert result == {'good': 0, 'is': 1, 'java': 2}


def test_normalize_negative_one():
    data = [10,1]
    out = normalize_negative_one(data)
    assert np.array_equal(out , np.array([1,-1]))
