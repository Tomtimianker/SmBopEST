import argparse
import bisect
# from anytree.search import *
import copy
# import gc
import gc
import inspect
import itertools
import json
import logging
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from itertools import *
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (Attention, FeedForward, Seq2SeqEncoder,
                              Seq2VecEncoder, TextFieldEmbedder,
                              TimeDistributed)
from allennlp.modules.matrix_attention.bilinear_matrix_attention import \
    BilinearMatrixAttention
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
# from torch_geometric.data import Data, Batch
from allennlp.nn import util
from allennlp.nn.util import min_value_of_dtype, replace_masked_values
# from modules.gated_graph_conv import GatedGraphConv
from allennlp.training.metrics import Average
from anytree import LevelOrderGroupIter, Node, PostOrderIter
from torch.nn.modules.rnn import LSTM, LSTMCell
from torch.utils import dlpack

# from utils.logger as logger
# from utils import logger
from utils import node_util, utils
# from utils.convert import *

# from random import shuffle
# from joblib import Parallel, delayed
# from joblib import parallel_backend
# from modules.tree_lstm import TreeLSTM, TreeLSTMv2, LSTMCellv2, Scalar
# from overrides import overrides





# from utils import indexing



def shuffle(t):
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx].view(t.size())





def new_isin(key,query):
    key,_ = key.sort()
    a= torch.searchsorted(key,query,right=True)
    b= torch.searchsorted(key,query ,right=False)
    return (a!=b).float()

def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))

def get_span_scores(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.
    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
#     best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
#     span_start_indices = best_spans // passage_length
#     span_end_indices = best_spans % passage_length
#     return torch.stack([span_start_indices, span_end_indices], dim=-1)
    return valid_span_log_probs




def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c", "--config", metavar="C", default="None", help="The Configuration file"
    )
    args = argparser.parse_args()
    return args


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout


import torch


@contextmanager
def torch_full_print():
    torch.set_printoptions(profile="full")
    try:
        yield None
    finally:
        torch.set_printoptions(profile="default")


@contextmanager
def stdout_to_file(filename, m="a"):
    with open(filename, m) as new_stdout:
        save_stdout = sys.stdout
        sys.stdout = new_stdout
        try:
            yield None
        finally:
            sys.stdout = save_stdout


def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


def get_total_alloc():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # obj.size()
                # self.logger.debug(f"{type(obj)}, {obj.size()},{find_names(obj)}")
                total += reduce(lambda x, y: x * y, obj.size()) * obj.element_size()

        except:
            pass
    return total


# import pstats
# with open("profile_t4.txt", "w") as f:
#     with stdout_redirected(f):
#         p = pstats.Stats('profile_t2.txt')
#         p.strip_dirs().sort_stats(-1).print_stats()
