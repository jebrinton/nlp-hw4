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

import pickle


class SP(Base):
    def __init__(self: Type["SP"]) -> None:
        super().__init__()

    def _init_tm(self: Type["SP"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        super()._init_tm(word_corpus, tag_corpus, init_val=init_val)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        

    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


    def sp_training_algorithm(self: Type["SP"],
                              word_corpus: Sequence[Sequence[str]],
                              tag_corpus: Sequence[Sequence[str]],
                              ) -> Tuple[int, int]:
        # TODO: complete me!
        #    This method should implement the structured perceptron training algorithm,
        #    and it should return a pair of ints (num_correct, num_total).
        #       num_correct should contain the number of tags that were correctly predicted
        #       num_total should contain the number of total tags predicted
        '''Initialize bigram tag model with 0s and emission model with 0s
        For each training sequence 𝒘, 𝒕
        Tag 𝒘 to get predicted tag sequence 𝒑=𝑝_1,𝑝_2, …, 𝑝_𝑛
        For 𝑖=1…𝑛
        Increment bigram entry for (𝑡_(𝑖−1),𝑡_𝑖 )
        Decrement bigram entry for (𝑝_(𝑖−1),𝑝_𝑖 )

        Increment emission entry for (𝑡_𝑖,𝑤_𝑖 )
        Decrement emission entry for (𝑝_𝑖,𝑤_𝑖 )
        Increment bigram entry for (𝑡_𝑛,<𝐸𝑂𝑆>)
        Decrement bigram entry for (𝑝_𝑛,<𝐸𝑂𝑆>)
        '''

        num_correct = 0
        num_total = 0

        # For each training sequence 𝒘, 𝒕
        for i, predicted_tags in enumerate(self.predict(word_corpus)):
            sentence = word_corpus[i]
            true_tags = tag_corpus[i]

            prev_pred_tag = START_TOKEN
            prev_true_tag = START_TOKEN

            for word, pred_tag, true_tag in zip(sentence, predicted_tags, true_tags):
                self.lm.increment_value(prev_pred_tag, pred_tag, -1)
                self.lm.increment_value(prev_true_tag, true_tag, 1)

                prev_pred_tag = pred_tag
                prev_true_tag = true_tag
                
                self.tm.increment_value(pred_tag, word, -1)
                self.tm.increment_value(true_tag, word, 1)

                if pred_tag == true_tag:
                    num_correct += 1
                
            self.lm.increment_value(prev_pred_tag, END_TOKEN, -1)
            self.lm.increment_value(prev_true_tag, END_TOKEN, 1)

            num_total += len(sentence)

        return (num_correct, num_total)


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
            transition_prob = 0

            # print(lgraph.node_layers[-1][child_tag])

            # if self.lm.get_value(parent_tag, child_tag) > 0 and self.tm.get_value(child_tag, word) > 0:
            transition_prob = log_value + self.lm.get_value(parent_tag, child_tag) + self.tm.get_value(child_tag, word)
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

