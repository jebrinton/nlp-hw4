# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Type, Tuple
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
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class HMM(Base):
    def __init__(self: Type["HMM"]) -> None:
        super().__init__()

    def _train_lm(self: Type["HMM"],
                  tag_corpus: Sequence[Sequence[str]]
                  ) -> None:
        self.lm_count_bigram(tag_corpus)
        self.lm.normalize_cond()

    def _train_tm(self: Type["HMM"],
                  word_corpus: Sequence[Sequence[str]],
                  tag_corpus: Sequence[Sequence[str]]
                  ) -> None:
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            for w, t in zip(word_seq, tag_seq):
                self.tm.increment_value(t, w, val=1)
            self.tm.increment_value(END_TOKEN, END_TOKEN, val=1)
        self.tm.normalize_cond(add=0.1)

    def _train(self: Type["HMM"],
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]]
               ) -> Type["HMM"]:
        super()._train(word_corpus, tag_corpus)
        self._train_lm(tag_corpus)
        self._train_tm(word_corpus, tag_corpus)
        return self

    def viterbi(self: Type["HMM"],
                word_list) -> Tuple[Sequence[str], float]:

        # TODO: complete me!
        # this method should return the most probable path along with the logprob of the most probable path
        return (["string", "string2"], -0.01991)
