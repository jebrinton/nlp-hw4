# SYSTEM IMPORTS
from collections.abc import Callable, Sequence
from typing import Tuple, Type
from tqdm import tqdm
import numpy as np
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from base import Base
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class SP(Base):
    def __init__(self: Type["SP"]) -> None:
        super().__init__()

    def _init_tm(self: Type["SP"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        super()._init_tm(word_corpus, tag_corpus, init_val=init_val)

    def sp_training_algorithm(self: Type["SP"],
                              word_corpus: Sequence[Sequence[str]],
                              tag_corpus: Sequence[Sequence[str]],
                              ) -> Tuple[int, int]:
        # TODO: complet me!
        #    This method should implement the structured perceptron training algorithm,
        #    and it should return a pair of ints (num_correct, num_total).
        #       num_correct should contain the number of tags that were correctly predicted
        #       num_total should contain the number of total tags predicted

    def _train(self: Type["SP"],
               train_word_corpus: Sequence[Sequence[str]],
               train_tag_corpus: Sequence[Sequence[str]],
               dev_word_corpus: Sequence[Sequence[str]] = None,
               dev_tag_corpus: Sequence[Sequence[str]] = None,
               max_epochs: int = 20,
               converge_error: float = 1e-4,
               log_function: Callable[[Type["SP"], int, Tuple[int, int], Tuple[int, int]], None] = None
               ) -> Type["SP"]:
        super()._train(train_word_corpus, train_tag_corpus)

        current_epoch: int = 0
        current_accuracy: float = 1.0
        prev_accuracy: float = 1.0
        percent_rel_error: float = 1.0

        while current_epoch < max_epochs and percent_rel_error > converge_error:

            train_correct, train_total = self.sp_training_algorithm(train_word_corpus, train_tag_corpus)
            dev_correct, dev_total = 0, 0

            if dev_word_corpus is not None and dev_tag_corpus is not None:

                for i, predicted_tags in enumerate(self.predict(dev_word_corpus)):
                    true_tags = dev_tag_corpus[i]
                    dev_total += len(true_tags)
                    dev_correct += np.sum(np.array(true_tags) == np.array(predicted_tags))

            if log_function is not None:
                log_function(self, current_epoch, (train_correct, train_total), (dev_correct, dev_total))

            epoch_correct = train_correct if dev_word_corpus is None or dev_tag_corpus is None else dev_correct
            epoch_total = train_total if dev_word_corpus is None or dev_tag_corpus is None else dev_total

            prev_accuracy = current_accuracy
            current_accuracy = float(epoch_correct) / float(epoch_total)
            percent_rel_error = abs(prev_accuracy - current_accuracy) / prev_accuracy

            current_epoch += 1

        return self

    def train_from_raw(self: Type["SP"],
                       train_word_corpus: Sequence[Sequence[str]],
                       train_tag_corpus: Sequence[Sequence[str]],
                       dev_word_corpus: Sequence[Sequence[str]] = None,
                       dev_tag_corpus: Sequence[Sequence[str]] = None,
                       max_epochs: int = 20,
                       converge_error: float = 1e-4,
                       log_function: Callable[[int, Tuple[int, int], Tuple[int, int]], None] = None
                       ) -> None:
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    def train_from_file(self: Type["SP"],
                        train_path: str,
                        dev_path: str = None,
                        max_epochs=20,
                        converge_error: float = 1e-4,
                        limit: int = -1
                        ) -> None:
        train_word_corpus, train_tag_corpus = load_annotated_data(train_path, limit=limit)
        dev_word_corpus, dev_tag_corpus = None, None
        if dev_path is not None:
            dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_path, limit=limit)
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    def viterbi(self: Type["SP"],
                word_list: Sequence[str]
                ) -> Tuple[Sequence[str], float]:

        # TODO: complete me!
        # This method should look identical to your HMM viterbi!

