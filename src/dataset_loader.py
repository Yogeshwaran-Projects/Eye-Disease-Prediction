import os
import pandas as pd
from sklearn.model_selection import train_test_split

class EyeDiseaseDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def dataPaths(self):
        filepaths, labels = [], []
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)
        return filepaths, labels

    def dataFrame(self, filepaths, labels):
        return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

    def split_(self):
        filepaths, labels = self.dataPaths()
        df = self.dataFrame(filepaths, labels)
        train, test_val = train_test_split(df, train_size=0.8, stratify=df['labels'], random_state=42)
        valid, test = train_test_split(test_val, train_size=0.5, stratify=test_val['labels'], random_state=42)
        return train, valid, test
