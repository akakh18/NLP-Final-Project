import re
from typing import List

from gensim.models import Word2Vec

INIT_DATA = [["გამარჯობა", "როგორ", "ხარ?"], ["რავი", "კარგად", "შენ?"]]

CHECK_NON_EMPTY_FN = lambda s: len(s) > 0


class GeorgianWord2Vec:
    def __init__(self) -> None:
        print("Initializing data")
        self.__model_name = "word2vec.model"
        model = Word2Vec(sentences=INIT_DATA, vector_size=100, window=5, min_count=1, workers=4, epochs=3)
        model.save(self.__model_name)
        self.sentences_split_regex = "\. |!|\?"

    # Returns word2vec model, trained by initial dataset
    def get_model(self) -> Word2Vec:
        return Word2Vec.load(self.__model_name)

    # Gets the file path and trains the model using its sentences
    def train(self, file_path: str) -> None:
        model = self.get_model()
        sentences = self.__convert_file_into_input(file_path)
        print(sentences[0])
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save(self.__model_name)

    # Returns vector corresponding to passed word
    def get_vector(self, word: str) -> List[int]:
        return Word2Vec.load(self.__model_name).wv[word]

    def __convert_file_into_input(self, file_path: str) -> List[List[str]]:
        with open(file_path, 'r') as f:
            data = f.read()
        sentences = re.split(self.sentences_split_regex, data)

        return list(
            filter(CHECK_NON_EMPTY_FN,
                   [list(filter(CHECK_NON_EMPTY_FN,
                                [word.strip() for word in sentence.split(" ")])) for sentence in sentences]))
