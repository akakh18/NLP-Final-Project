import re
from typing import List

from gensim.models import Word2Vec

INIT_DATA = [["გამარჯობა", "როგორ", "ხარ?"], ["რავი", "კარგად", "შენ?"]]

CHECK_NON_EMPTY_FN = lambda s: len(s) > 0


class GeorgianWord2Vec:
    def __init__(self) -> None:
        self.model = Word2Vec(sentences=INIT_DATA, vector_size=100, window=5, min_count=1, workers=4)
        self.sentences_split_regex = "\. |!|\?"

    # Returns word2vec model, trained by initial dataset
    def get_model(self) -> Word2Vec:
        return self.model

    # Gets the file path and trains the model using its sentences
    def train(self, file_path: str) -> None:
        sentences = self.__convert_file_into_input(file_path)
        self.model.train(sentences, self.model.corpus_count, self.model.epochs)

    # Returns vector corresponding to passed word
    def get_vector(self, word: str) -> List[int]:
        return self.model.wv[word]

    def __convert_file_into_input(self, file_path: str) -> List[List[str]]:
        with open(file_path, 'r') as f:
            data = f.read()
        sentences = re.split(self.sentences_split_regex, data)

        return list(
            filter(CHECK_NON_EMPTY_FN,
                   [list(filter(CHECK_NON_EMPTY_FN,
                                [word.strip() for word in sentence.split(" ")])) for sentence in sentences]))
