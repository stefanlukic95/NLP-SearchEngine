from typing import Sequence
import pandas as pd
import numpy as np
from numpy.linalg import norm


# utility function for nomalization between [-1,1]
def normalize_negative_one(data):
    normalized_input = (data - np.min(data)) / (np.max(data) - np.min(data))
    return 2 * normalized_input - 1


class DataPreprocessing:

    def __init__(self, data_path):
        self.data_path = data_path


    def load_data(self, data_path):
        df = pd.read_json(data_path, lines=True)
        return df

    def clean_data(self, df):
        df['question'] = df['question'].str.replace(r'[^\w\s]+', '')
        df['question'] = df['question'].str.lower()
        return df['question']


