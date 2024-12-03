# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Type, Tuple
import numpy as np
import os
import sys
import math


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
        lgraph = LayeredGraph(init_val = -np.inf)

        def init_func() -> LayeredGraph:
            return lgraph

        def update_func_ptr(child_tag : str,
                     word : str,
                     parent_tag : str,
                     log_value : float,
                     lgraph : LayeredGraph
                     ) -> None:

            # tm is tag to word
            # lm is parent to child
            transition_prob = -math.inf

            # print(lgraph.node_layers[-1][child_tag])

            if self.lm.get_value(parent_tag, child_tag) > 0 and self.tm.get_value(child_tag, word) > 0:
                transition_prob = log_value + np.log(self.lm.get_value(parent_tag, child_tag)) + np.log(self.tm.get_value(child_tag, word))
                # print("tp")
                # print(transition_prob)
                # print(self.lm.get_value(parent_tag, child_tag))
                # print(self.tm.get_value(child_tag, word))

            # print("get node in layer")
            # print(lgraph.get_node_in_layer(child_tag)[0])

            if (transition_prob > lgraph.get_node_in_layer(child_tag)[0]):
                lgraph.add_node(child_tag, transition_prob, parent_tag)
            
        self.viterbi_traverse(word_list, init_func, update_func_ptr)

        path = list([END_TOKEN])

        # try with -1, 0, -1 if you must
        for i in range(len(word_list)+1, 1, -1):
            parent_tag = lgraph.node_layers[i][path[0]][1]
            path.insert(0, parent_tag)

        # remove BOS from path, return the logprob of EOS
        return path[:-1], lgraph.node_layers[-1][END_TOKEN][0]

        # only change for sp.py:
        # line 70 isn't there
        # don't need math.log (we are dealing with counts)

