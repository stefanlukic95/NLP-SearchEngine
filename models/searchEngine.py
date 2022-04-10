from typing import Sequence
from data_preprocessing.TfidVectorize import TfidVectorize
from data_preprocessing.preprocessing import DataPreprocessing


class QuestionsSearchEngine:
    def __init__(self, questions: Sequence[str], data_path) -> None:
        self.questions = questions
        self.data_path = data_path

    def most_similar(self, query: str):
        vectorizer = TfidVectorize(self.questions)
        fit_questions = vectorizer.fit_custom(self.questions)
        query_transform = vectorizer.transform_custom(query, fit_questions)
        question_transfrom = vectorizer.transform_custom(self.questions, fit_questions)

        cosine_similarity_res, top_results, top_indices = vectorizer.cosine_similarity(query_transform,
                                                                                       question_transfrom)
        data_preprocessing = DataPreprocessing(self.data_path)
        verbatim_data = data_preprocessing.load_data(self.data_path)
        for i in range(5):
            index = top_indices[i]
            print("Question:", verbatim_data['question'][index], "\t\t\tID:", verbatim_data['id'][index],
                  "\t\t\t\t\t\t\tSimilarity_score:", top_results[i])
