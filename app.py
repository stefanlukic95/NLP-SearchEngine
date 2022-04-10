import os
from data_preprocessing.preprocessing import DataPreprocessing
from models.searchEngine import QuestionsSearchEngine


def main():
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, 'data/questions.jsonl')
    data_preprocessing = DataPreprocessing(data_path)

    # load and clean data
    load_corpus = data_preprocessing.load_data(data_path)
    clean_corpus = data_preprocessing.clean_data(load_corpus)
    clean_corpus = clean_corpus[:1000]

    query_question = [input("Enter a question:")]
    engine = QuestionsSearchEngine(clean_corpus, data_path)
    engine.most_similar(query_question)


if __name__ == "__main__":
    main()
