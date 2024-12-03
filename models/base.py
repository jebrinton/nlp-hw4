# SYSTEM IMPORTS
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
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
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import BigramTable, EmissionTable, START_TOKEN, END_TOKEN, UNK_TOKEN


class Base(object):
    def __init__(self: Type["Base"]):
        self.lm: BigramTable = None
        self.tm: EmissionTable = None
        self.tag_vocab: Set[str] = set()
        self.word_vocab: Set[str] = set()

    def _init_lm(self: Type["Base"],
                 tag_corpus: Sequence[Sequence["str"]],
                 init_val: float = 0.0
                 ) -> None:
        # two main differences for the alphabets of the tag model:
        #     1) The input alphabet contains the START symbol
        #     2) The output alphabet contains the STOP symbol
        tag_vocab = set()
        for tag_seq in tag_corpus:
            tag_vocab.update(tag_seq)

        self.tag_vocab = tag_vocab
        sorted_tags = sorted(tag_vocab)

        # personally I like to keep the alphabet sorted b/c it helps me read the table if I have to
        self.lm = BigramTable(sorted_tags, sorted_tags, init_val=init_val)

    def _init_tm(self: Type["Base"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        word_vocab = set()
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            word_vocab.update(word_seq)

        # add UNK to the words
        self.word_vocab = word_vocab | set([UNK_TOKEN])

        # again I prefer to keep things sorted in case I need to try and reproduce something from lecture
        # when I'm debugging
        sorted_words = sorted(self.word_vocab)
        sorted_tags = sorted(self.tag_vocab)
        self.tm = EmissionTable(sorted_tags, sorted_words, init_val=init_val)

    def lm_count_bigram(self: Type["Base"],
                        tag_corpus: Sequence[Sequence[str]]
                        ) -> None:
        # TODO: complete me!
        #   iterate through each sequence of the corpus and increment the corresponding bigram entries
        #   don't forget to increment <EOS> after each sequence!
        for seq in tag_corpus:
            for prev, cur in zip([START_TOKEN] + list(seq), list(seq) + [END_TOKEN]):
                self.lm.increment_value(prev, cur)
        return None

    def _train(self: Type["Base"],
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]],
               init_val: float = 0.0
               ) -> Type["Base"]:
        self._init_lm(tag_corpus, init_val=init_val)
        self._init_tm(word_corpus, tag_corpus, init_val=init_val)
        return self

    def train_from_raw(self: Type["Base"],
                       word_corpus: Sequence[Sequence[str]],
                       tag_corpus: Sequence[Sequence[str]],
                       limit: int = -1
                       ) -> Type["Base"]:
        if limit > -1:
            word_corpus = word_corpus[:limit]
            tag_corpus = tag_corpus[:limit]
        return self._train(word_corpus, tag_corpus)

    def train_from_file(self: Type["Base"],
                        file_path: str,
                        limit: int = -1
                        ) -> Type["Base"]:
        word_corpus, tag_corpus = load_annotated_data(file_path, limit=limit)
        return self._train_from_raw(word_corpus, tag_corpus, limit=limit)

    def parse_word_list(self: Type["Base"],
                        word_list: Sequence[str]
                        ) -> Sequence[str]:
        parsed_list: Sequence[str] = list(word_list)
        for i, w in enumerate(parsed_list):
            if w not in self.word_vocab:
                parsed_list[i] = UNK_TOKEN
        return parsed_list

    def viterbi_traverse(self: Type["Base"],
                         word_list: Sequence[str],
                         init_func_ptr: Callable[[], LayeredGraph],
                         update_func_ptr: Callable[[str, str, str, float, LayeredGraph], None],
                         ) -> None:
        # TODO: complete me!
        # This function should implement the viterbi traversal on a LayeredGraph object
        #   an initialized LayeredGraph object will be produced by init_func_ptr
        #   and your code should populate this graph. To be clear, the traversal code here will
        #   need to allocate new layers on the graph, but to populate the newly created layer,
        #   you can call update_func_ptr
        lgraph : LayeredGraph = init_func_ptr()

        lgraph.add_layer()
        lgraph.add_node(START_TOKEN, 0.0, None)
        
        # exit()
        # print(lgraph.node_layers[-1].items())

        # each iteration of this populates a new layer
        for word in word_list:
            lgraph.add_layer()

            # print('parentslayeritems')
            # print(lgraph.node_layers[-2].items())

            # iterate thru parents, then thru possible child tags
            for parent_tag, (pathcost, _) in lgraph.node_layers[-2].items():
                for child_tag in self.tag_vocab - set([UNK_TOKEN]):
                    update_func_ptr(child_tag, word, parent_tag, pathcost, lgraph)


        lgraph.add_layer()
        for parent_tag, [pathcost, _] in lgraph.node_layers[-2].items():
            update_func_ptr(END_TOKEN, END_TOKEN, parent_tag, pathcost, lgraph)

    def predict_sentence(self: Type["Base"],
                         word_list: Sequence[str]
                         ) -> Sequence[str]:
        word_list = self.parse_word_list(word_list)
        path, log_prob = self.viterbi(word_list)
        return path

    def predict(self: Type["Base"],
                word_corpus: Sequence[Sequence[str]]
                ) -> Iterable[Sequence[str]]:
        for word_list in word_corpus:
            yield self.predict_sentence(word_list)

