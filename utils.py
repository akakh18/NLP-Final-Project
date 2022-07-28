from typing import List, Tuple
import os.path
from typing import Protocol, List

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nn
from torch.types import Device
from torchtext.vocab import build_vocab_from_iterator, Vocab
from gensim.models import Word2Vec, FastText
from nltk.tokenize import sent_tokenize

INIT_DATA = [["გამარჯობა", "როგორ", "ხარ?"], ["რავი", "კარგად", "შენ?"]]


def strip_word(word: str) -> str:
    r_strip_chars = ['.', ',', '!', '?', ':', ';']

    result = word

    for c in r_strip_chars:
        result = result.rstrip(c)

    return result.strip().strip("(").strip(")")


def convert_file_into_input(file_path: str) -> List[List[str]]:
    with open(file_path, 'r') as f:
        data = f.read()

    sentences = sent_tokenize(data)
    return [[strip_word(word) for word in sentence.split(" ")] for sentence in sentences]


class GeorgianModel(Protocol):
    def get_model(self) -> Word2Vec:
        """Returns word2vec model, trained by initial dataset"""

    def train(self, file_path: str) -> None:
        """Gets the file path and trains the model using its sentences"""

    def get_vector(self, word: str) -> List[int]:
        """Returns vector corresponding to passed word"""


class GeorgianWord2VecModel:
    def __init__(self, load: bool = False) -> None:
        print("Initializing data")
        self.__model_name = "word2vec.model"
        if not load or not os.path.exists(self.__model_name):
            model = Word2Vec(sentences=INIT_DATA, vector_size=100, window=5, min_count=1, workers=4, epochs=3)
            model.save(self.__model_name)
        print("Model created!")

    def get_model(self) -> Word2Vec:
        return Word2Vec.load(self.__model_name)

    def train(self, file_path: str) -> None:
        model = self.get_model()
        sentences = convert_file_into_input(file_path)
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(self.__model_name)

    def get_vector(self, word: str) -> List[int]:
        return Word2Vec.load(self.__model_name).wv.get_vector(word)


class GeorgianFastTextModel:
    def __init__(self, load: bool = False) -> None:
        print("Initializing data")
        self.__model_name = "fasttext.model"
        if not load or not os.path.exists(self.__model_name):
            model = FastText(sentences=INIT_DATA, vector_size=100, window=5, min_count=1, workers=4, epochs=3)
            model.save(self.__model_name)
        print("Model created!")

    def get_model(self) -> FastText:
        return FastText.load(self.__model_name)

    def train(self, file_path: str) -> None:
        model = self.get_model()
        sentences = convert_file_into_input(file_path)
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(self.__model_name)

    def get_vector(self, word: str) -> List[int]:
        return FastText.load(self.__model_name).wv.get_vector(word)
