import re
import pandas as pd

from sklearn.model_selection import train_test_split


def load_dataset(path, random_seed):
    data = pd.read_csv(path)

    X = data['content']
    y = data['label']

    return train_test_split(X, y, test_size=0.3, random_state=random_seed)


def escape_spec(string):
    return re.sub('[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', '', string)
