import pandas as pd
from torch.utils.data.dataset import Dataset
import json
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        with open(data_path) as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                js = json.loads(line)
                label = int(js["stars"]) - 1
                texts.append(js["text"])
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        with Pool(30) as p:
            self.document_encode = [x for x in p.map(self.get_code, range(len(self.labels)))]
            
    def get_code(self, index):
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                        :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        return document_encode
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        return self.document_encode[index].astype(np.int64), label
