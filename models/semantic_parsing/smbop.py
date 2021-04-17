from typing import Dict, List, Tuple
import itertools
from functools import lru_cache
import pandas as pd
import os
import torch
import allennlp
from allennlp.nn.util import masked_mean
from allennlp.common.util import *
import  allennlp.nn.util as ai2_util
from allennlp.data import Vocabulary
from allennlp.models import Model

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder,ClsPooler
from allennlp.modules.matrix_attention.bilinear_matrix_attention import (
    BilinearMatrixAttention,
)
from allennlp.modules import (
    TextFieldEmbedder,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    Attention,
    FeedForward,
    TimeDistributed,
)

from allennlp.nn import util
from allennlp.data import TokenIndexer

from allennlp.training.metrics import Average, F1Measure
from copy import deepcopy
from collections import Counter, defaultdict, OrderedDict
import bisect
import numpy as np
import random

from overrides import overrides

import torch
import numpy as np
import pandas as pd
from anytree import Node, PostOrderIter, LevelOrderGroupIter


from utils import node_util
from utils import utils as frontier_utils

import json

from eval_final.evaluation import evaluate_single

import gc, inspect
import logging
import itertools
import numpy as np
import torch

import utils.node_util as node_util
import numpy as np
from functools import partial
import torch
import allennlp.nn.util as util
import time
logger = logging.getLogger(__name__)

def get_span_indices(is_gold_span,Kdiv2 = 10):
    # device = is_gold_span.device
    batch_size = is_gold_span.size(0)
    is_gold_span_list = is_gold_span.nonzero().tolist()
    l_start = [[] for _ in range(batch_size)]
    l_end = [[] for _ in range(batch_size)]
    for el in is_gold_span_list:
        b,s,e = el
        l_start[b].append(s)
        l_end[b].append(e)
    for l in l_start:
        l.extend([-1]*(Kdiv2-len(l)))
    for l in l_end:
        l.extend([-1]*(Kdiv2-len(l)))
    l_end = torch.tensor(l_end,dtype=is_gold_span.dtype,device=is_gold_span.device)
    l_start = torch.tensor(l_start,dtype=is_gold_span.dtype,device=is_gold_span.device)
    return l_start,l_end

def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    It does not guarantee element order within each span.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = util.get_range_vector(
        max_batch_span_width, util.get_device_of(target)
    ).view(1, 1, -1)
    #     print(max_batch_span_width)
    #     print(max_span_range_indices)
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    #     raw_span_indices = span_ends - max_span_range_indices
    raw_span_indices = span_starts + max_span_range_indices
    #     print(raw_span_indices)
    #     print(target.size())
    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the sequence
    # have an end index < max_batch_span_width, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1))
    #     print(span_mask)
    #     span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    span_indices = raw_span_indices * span_mask
    #     print(span_indices)

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = util.batched_index_select(target, span_indices)

    return span_embeddings, span_mask

def get_failed_set(agenda_hash,decoding_step,tree_obj,batch_size,hash_gold_levelorder):
    failed_set = []
    failed_list = []
    node_list = []
    for b in range(batch_size):
        node_list.append(node_util.print_tree(tree_obj[b]))
        node_dict  = {node.hash:node for node in PostOrderIter(tree_obj[b])}
        batch_set = (set(hash_gold_levelorder[b][decoding_step+1].tolist())-set(agenda_hash[b].tolist()))-{-1}
        failed_list.append([node_dict[set_el] for set_el in batch_set])
        failed_set.extend([node_dict[set_el] for set_el in batch_set])
    return failed_list,node_list,failed_set

def get_str_list(tokenizer,enc,batch_size,span_start_indices,span_end_indices):
    str_list = []
    for batch_el in range(batch_size):
        seq = enc['tokens']['token_ids'][batch_el][1:]
        idx = [[a,b] for a,b in zip(span_start_indices[batch_el].tolist(),span_end_indices[batch_el].tolist())]
        str_list.append([tokenizer.decode(seq[a:b+1].tolist()) for a,b in idx])
    return str_list



@lru_cache(maxsize=128)
def compute_op_idx(batch_size,seq_len,binary_op_count,unary_op_count,device):
    binary_op_ids = torch.arange(binary_op_count,dtype=torch.int64,device=device).expand(
        [batch_size, seq_len ** 2, binary_op_count]
    )
    unary_op_ids = (
        torch.arange(unary_op_count,dtype=torch.int64,device=device) + binary_op_count
    ).expand([batch_size, seq_len, unary_op_count])
    
    frontier_op_ids = torch.cat(
        [
            binary_op_ids.reshape([batch_size, -1]),
            unary_op_ids.reshape([batch_size, -1]),
        ],
        dim=-1,
    )
    return frontier_op_ids


@lru_cache(maxsize=128)
def compute_agenda_idx(batch_size,seq_len,binary_op_count,unary_op_count,device ):
    binary_agenda_idx = (
        torch.arange(seq_len ** 2,device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([batch_size, seq_len ** 2, binary_op_count])
        .reshape([batch_size, -1])
    )
    l_binary_agenda_idx = binary_agenda_idx // seq_len
    r_binary_agenda_idx = binary_agenda_idx % seq_len
    unary_agenda_idx = (
        torch.arange(seq_len,device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([batch_size, seq_len, unary_op_count])
        .reshape([batch_size, -1])
    )
    l_agenda_idx = torch.cat([l_binary_agenda_idx, unary_agenda_idx], dim=-1)
    r_agenda_idx = torch.cat([r_binary_agenda_idx, unary_agenda_idx], dim=-1)
    return l_agenda_idx, r_agenda_idx



class Item:
    def __init__(self,curr_type,l_child_idx,r_child_idx,mask):
        self.curr_type = curr_type
        self.l_child_idx = l_child_idx
        self.r_child_idx = r_child_idx
        self.mask = mask

class ZeroItem:
    def __init__(self,curr_type,final_leaf_indices,span_start_indices,span_end_indices,entities,enc,tokenizer):
        self.curr_type = curr_type
        self.final_leaf_indices = final_leaf_indices
        self.span_start_indices = span_start_indices
        self.span_end_indices = span_end_indices
        self.entities = entities
        self.enc = enc
        self.tokenizer = tokenizer


@Model.register("smbop_parser")
class SmbopParser(Model):
    def __init__(
        self,
        experiment_name: str,
        vocab: Vocabulary,
        question_embedder: TextFieldEmbedder,
        schema_encoder: Seq2SeqEncoder,
        agenda_encoder: Seq2SeqEncoder,
        # ranker_contextualizer: Seq2SeqEncoder,
        tree_rep_transformer: Seq2SeqEncoder,
        utterance_augmenter: Seq2SeqEncoder,
        agenda_summarizer: Seq2SeqEncoder,
        decoder_timesteps=9,
        agenda_size=30,
        d_frontier = 50,
        misc_params=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)
        self._experiment_name = experiment_name
        self._misc_params = misc_params
        self.set_flags()
        self._utterance_augmenter = utterance_augmenter
        self._action_dim = agenda_encoder.get_output_dim()
        self.d_frontier = d_frontier
        self._agenda_size = agenda_size
        self.n_schema_leafs = 15

        self.tokenizer = TokenIndexer.by_name("pretrained_transformer")(model_name="Salesforce/grappa_large_jnt")._allennlp_tokenizer.tokenizer

        if not self.cntx_reranker:
            self._noreranker_cntx_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2*self._action_dim
            )
        if not self.cntx_beam:
            self._nobeam_cntx_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2*self._action_dim
            )
        self.activation_func = torch.nn.ReLU
        # else:
        if self.lin_after_cntx:
            self.cntx_linear = torch.nn.Sequential(
            torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(4*self._action_dim),
            self.activation_func(),
            torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
        )
        if self.cntx_rep:
            self._cntx_rep_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2*self._action_dim
            )
        # self.use_multint = True
        # self.multint = MultInt(input_dim=2*self._action_dim,n_head=8)
        self.use_add = True
        self._create_action_dicts()
        self.op_count = self.binary_op_count + self.unary_op_count
        self.xent = torch.nn.CrossEntropyLoss()
        

        

        # self.activation_func = torch.nn.GELU
        self.type_embedding = torch.nn.Embedding(self.op_count,self._action_dim)
        self.summrize_vec = torch.nn.Embedding(num_embeddings=1,embedding_dim=self._action_dim)
        self._binary_frontier_embedder = BilinearMatrixAttention(
            2*self._action_dim, 2*self._action_dim, label_dim=self.d_frontier
        )
        self.max_norm = 0 
        if self.use_add:
            self.d_frontier = 2*self._action_dim
            self.left_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
            )
            self.right_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
            )
            self.after_add  = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            )
            self._unary_frontier_embedder  = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            )
        else:
            self._unary_frontier_embedder = torch.nn.Linear(
                in_features=2*self._action_dim, out_features=self.d_frontier
            )
        self.op_linear = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.op_count
        )
        self.pre_op_linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            # torch.nn.Linear(self.d_frontier, self.d_frontier),
            self.activation_func(),
        )
        # self._score_depth  = torch.nn.Sequential(
        #     torch.nn.Linear(self._action_dim, 2*self._action_dim),
        #     torch.nn.Dropout(p=dropout),
        #     torch.nn.LayerNorm(2*self._action_dim),
        #     self.activation_func(),
        #     torch.nn.Linear(2*self._action_dim, 3+decoder_timesteps),
        #     )
        # self._score_depth  = torch.nn.Sequential(
        #     torch.nn.Linear(self._action_dim, 2*self._action_dim),
        #     torch.nn.Dropout(p=dropout),
        #     torch.nn.LayerNorm(2*self._action_dim),
        #     self.activation_func(),
        #     torch.nn.Linear(2*self._action_dim, 1),
        #     )

        #old
        assert ((self._action_dim%2)==0)
        self.vocab = vocab
        self._question_embedder = question_embedder
        self._schema_encoder = schema_encoder
        self._agenda_encoder = agenda_encoder
        # self._ranker_contextualizer = ranker_contextualizer
        self._agenda_summarizer = agenda_summarizer
        
        self._tree_rep_transformer = tree_rep_transformer
        # self.tree_lstm = TreeLSTM(self._action_dim // 2, dropout=dropout)
        
        self._decoder_timesteps = decoder_timesteps
        self._agenda_size = agenda_size
        self.q_emb_dim = question_embedder.get_output_dim()

        
        self.dropout_prob = dropout
        self._action_dim = agenda_encoder.get_output_dim()
        self._span_score_func = torch.nn.Linear(self._action_dim , 2)
        # self._pooler = ClsPooler(embedding_dim=self._action_dim)
        self._pooler = BagOfEmbeddingsEncoder(embedding_dim=self._action_dim)

        
        self._rank_schema = torch.nn.Sequential(
            torch.nn.Linear(self._action_dim, self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._action_dim, 1),
        )
        self._rank_agenda = torch.nn.Sequential(
            torch.nn.Linear(2*self._action_dim, 2*self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(2*self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(2*self._action_dim, 1),
        )
        self._emb_to_action_dim = torch.nn.Linear(
            in_features=self.q_emb_dim, out_features=self._action_dim,
        )

        self._create_type_tensor()
        
        self._bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        
        self._softmax = torch.nn.Softmax(dim=1)
        self._final_beam_acc = Average()
        self._reranker_acc = Average()
        self._spider_acc = Average()

        self._leafs_acc = Average()
        self._batch_size = -1
        self._device = None
        self._evaluate_func = partial(
            evaluate_single,
            db_dir=os.path.join("dataset", "database"),
            table_file=os.path.join("dataset", "scholar_table.json"),
        )


        

    def set_flags(self):
        print(self._misc_params)
        self.is_oracle = self._misc_params.get("is_oracle", False)
        self.ranking_ratio = self._misc_params.get("ranking_ratio", 0.7)
        self.unique_reranker = self._misc_params.get("unique_reranker", False)
        self.cntx_reranker = self._misc_params.get("cntx_reranker", True)
        self.lin_after_cntx = self._misc_params.get("lin_after_cntx", False)
        self.cntx_beam = self._misc_params.get("cntx_beam", True)
        self.cntx_rep = self._misc_params.get("cntx_rep", False)
        self.add_residual_beam = self._misc_params.get("add_residual_beam", False)
        self.add_residual_reranker = self._misc_params.get("add_residual_reranker", False)
        self.only_last_rerank = self._misc_params.get("only_last_rerank", False)
        self.oldlstm = self._misc_params.get("oldlstm", False)
        self.use_treelstm = self._misc_params.get("use_treelstm", False)
        self.disentangle_cntx = self._misc_params.get("disentangle_cntx", True)
        self.cntx_agenda =  self._misc_params.get("cntx_agenda", True)
        self.uniquify =  self._misc_params.get("uniquify", True)
        self.temperature = self._misc_params.get("temperature", 1.0)
        self.use_bce = self._misc_params['use_bce']
        self.value_pred = self._misc_params.get('value_pred',True)
        self.debug = self._misc_params.get('debug',False)
        # self.use_bce = True

        

        
        
        self.reuse_cntx_reranker = self._misc_params.get("reuse_cntx_reranker", True)
        self.should_rerank = self._misc_params.get("should_rerank", True)

    def _create_type_tensor(self):
        rule_tensor = [
            [[0] * len(self._type_dict) for _ in range(len(self._type_dict))]
            for _ in range(len(self._type_dict))
        ]
        # with open("utils/rules.json") as f:
        if self.value_pred:
            RULES = node_util.RULES_values
        else:
            RULES = node_util.RULES_novalues

        rules = json.loads(RULES)
        for rule in rules:
            i, j_k = rule
            if len(j_k) == 0:
                continue
            elif len(j_k) == 2:
                j, k = j_k
            else:
                j, k = j_k[0], j_k[0]
            try:
                i, j, k = self._type_dict[i], self._type_dict[j], self._type_dict[k]
            except:
                continue
            rule_tensor[i][j][k] = 1
        self._rule_tensor = torch.tensor(rule_tensor)
        self._rule_tensor[self._type_dict["keep"]] = 1
        self._rule_tensor_flat = self._rule_tensor.flatten()
        self._op_count = self._rule_tensor.size(0)

        self._term_ids = [
            self._type_dict[i]
            for i in [
                "Project",
                "Orderby_desc",
                "Limit",
                "Groupby",
                "intersect",
                "except",
                "union",
                "Orderby_asc",
            ]
        ]
        self._term_tensor = torch.tensor(
            [1 if i in self._term_ids else 0 for i in range(len(self._type_dict))]
        )

    def _create_action_dicts(self):
        unary_ops = [
            "keep",
            "min",
            "count",
            "max",
            "avg",
            "sum",
            "Subquery",
            "distinct",
            "literal",
        ]
        
        binary_ops = [
            "eq",
            "like",
            "nlike",
            "add",
            "sub",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
            "Orderby_desc",
            "Orderby_asc",
            "Project",
            "Selection",
            "Limit",
            "Groupby",
        ]
        self.binary_op_count = len(binary_ops)
        self.unary_op_count = len(unary_ops)
        self._op_names = [
            k for k in itertools.chain(binary_ops,unary_ops , ["nan", "Table", "Value"])
        ]
        self._type_dict = OrderedDict({k: i for i, k in enumerate(self._op_names)})
        self.keep_id = self._type_dict['keep']
        self._ACTIONS = {k: 1 for k in unary_ops}
        self._ACTIONS.update({k: 2 for k in binary_ops})
        self._ACTIONS = OrderedDict(self._ACTIONS)
        self._frontier_size = sum(
            self._agenda_size ** n for n in self._ACTIONS.values()
        )
        self.hasher = None
        self.flag_move_to_gpu = True

                
    def move_to_gpu(self,device):
        if self.flag_move_to_gpu:
            self._term_tensor = self._term_tensor.to(device)
            self._rule_tensor_flat = self._rule_tensor_flat.to(device)
            self._rule_tensor = self._rule_tensor.to(device)
            self.flag_move_to_gpu = False



    def forward(
        self,
        enc,
        db_id,
        leaf_hash,
        leaf_types,
        tree_obj=None,
        gold_sql=None,
        gold_subtrees=None,
        leaf_indices=None,
        world = None,
        entities=None,
        orig_entities=None,
        is_gold_leaf=None,
        lengths=None,
        offsets=None,
        relation=None,
        depth=None,
        hash_gold_levelorder=None,
        hash_gold_tree=None,
        span_hash=None,
        is_gold_span = None,
        
    ):
        
        total_start = time.time()
        outputs = {}
        agenda_list = []
        item_list = []
        self._device = enc["tokens"]["token_ids"].device
        self.move_to_gpu(self._device)
        batch_size = len(db_id)
        self.hasher = node_util.Hasher(self._device)
        (
            embedded_schema,
            schema_mask,
            embedded_utterance,
            utterance_mask,
        ) = self._encode_utt_schema(enc, offsets, relation, lengths)
        batch_size,utterance_length,_ = embedded_utterance.shape
        if depth is not None:
            depth = depth.sum(-1)
            depth_counter = depth.clone().unsqueeze(-1)
        start = time.time()
        loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        new_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        pre_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        depth_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        ranking_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        vector_loss = torch.tensor(
            [0] * batch_size, dtype=torch.float32, device=self._device
        )
        bce_vector_loss = torch.tensor(
            [0] * batch_size, dtype=torch.float32, device=self._device
        )
        # tree_sizes_vector = torch.tensor(
        #     [0] * batch_size, dtype=torch.float32, device=self._device
        # )

        tree_sizes_vector = torch.tensor(
            [1] * batch_size, dtype=torch.float32, device=self._device
        )
        if hash_gold_levelorder is not None:
            new_hash_gold_levelorder = hash_gold_levelorder.sort()[0].transpose(0,1)
        if self.value_pred:
            span_scores,start_logits,end_logits = self.score_spans(embedded_utterance,utterance_mask)
            span_mask = torch.isfinite(span_scores).bool()
            final_span_scores = span_scores.clone()
            # if span_hash is not None:
            delta = final_span_scores.shape[-1]-span_hash.shape[-1]
            span_hash = torch.nn.functional.pad(
                span_hash,
                pad=(0, delta,0,delta),
                mode="constant",
                value=-1,
            )
            if self.training:
                is_gold_span = torch.nn.functional.pad(is_gold_span,
                    pad=(0, delta,0,delta),
                    mode="constant",
                    value=0,
                )
                batch_idx,start_idx,end_idx = is_gold_span.nonzero().t()
                final_span_scores[batch_idx,start_idx,end_idx] = ai2_util.max_value_of_dtype(final_span_scores.dtype)
                #TODO : replace with the next line
                # final_span_scores = final_span_scores.masked_fill(is_gold_span,ai2_util.max_value_of_dtype(final_span_scores.dtype))
                # is_span_start = torch.zeros_like(utterance_mask,device=self._device,dtype=torch.float)
                # is_span_end = torch.zeros_like(utterance_mask,device=self._device,dtype=torch.float)
                # is_span_start[batch_idx,start_idx]=1
                # is_span_end[batch_idx,end_idx]=1
                
                is_span_end = is_gold_span.sum(-2).float()
                is_span_start = is_gold_span.sum(-1).float()
                
                span_start_probs = util.masked_log_softmax(start_logits, utterance_mask.bool(), dim=1)
                span_end_probs = util.masked_log_softmax(end_logits, utterance_mask.bool(), dim=1)
                
                vector_loss += (-span_start_probs*is_span_start.squeeze()).sum(-1) - (span_end_probs*is_span_end.squeeze()).sum(-1)
                tree_sizes_vector += 2*is_span_start.squeeze().sum(-1)

                
            else:
                final_span_scores = span_scores
            ten = 10
            # if not self.is_oracle:
                
            # else:
            #     ten = is_gold_span.sum(-1).float().sum(-1).int()
            top_input, leaf_span_mask, best_spans = util.masked_topk(final_span_scores.view([batch_size,-1]),span_mask.view([batch_size,-1]),ten)
            span_start_indices = best_spans // utterance_length
            span_end_indices = best_spans % utterance_length

                
                # assert is_gold_span is not None
                
                # span_start_indices, span_end_indices =  get_span_indices(is_gold_span,Kdiv2=ten)
                # leaf_span_mask = (span_start_indices>=0)
            


            start_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_start_indices
            )
            end_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_end_indices
            )
            span_rep = (end_span_rep +start_span_rep)/2
            leaf_span_rep = span_rep
            leaf_span_hash = allennlp.nn.util.batched_index_select(span_hash.reshape([batch_size,-1,1]),  best_spans).reshape([batch_size,-1])
            leaf_span_types = torch.where(leaf_span_mask,self._type_dict["Value"],self._type_dict["nan"]).int()

        leaf_schema_scores = self._rank_schema(embedded_schema)
        leaf_schema_scores = leaf_schema_scores/self.temperature
        if is_gold_leaf is not None:
            is_gold_leaf = torch.nn.functional.pad(
                is_gold_leaf,
                pad=(0, leaf_schema_scores.size(-2) - is_gold_leaf.size(-1)),
                mode="constant",
                value=0,
            )

        if self.training:
            final_leaf_schema_scores = leaf_schema_scores.clone()
            if not self.is_oracle:
                    if self.use_bce:
                        bce_vector_loss += (self._bce_loss(final_leaf_schema_scores,is_gold_leaf.unsqueeze(-1).float())*schema_mask.unsqueeze(-1).bool()).mean()
                    else:
                        avg_leaf_schema_scores = util.masked_log_softmax(final_leaf_schema_scores, schema_mask.unsqueeze(-1).bool(), dim=1)
                        loss_tensor = -avg_leaf_schema_scores * is_gold_leaf.unsqueeze(-1).float()
                        vector_loss += loss_tensor.squeeze().sum(-1)
                        tree_sizes_vector += is_gold_leaf.squeeze().sum(-1).float()

            final_leaf_schema_scores = final_leaf_schema_scores.masked_fill(
                is_gold_leaf.bool().unsqueeze(-1), ai2_util.max_value_of_dtype(final_leaf_schema_scores.dtype)
            )
        else:
            final_leaf_schema_scores = leaf_schema_scores
            
        final_leaf_schema_scores = final_leaf_schema_scores.masked_fill(
            ~schema_mask.bool().unsqueeze(-1), ai2_util.min_value_of_dtype(final_leaf_schema_scores.dtype)
        )
        
        min_k = torch.clamp(schema_mask.sum(-1),0,self.n_schema_leafs)
        _, leaf_schema_mask, top_agenda_indices = util.masked_topk(final_leaf_schema_scores.squeeze(-1),
                                                            mask=schema_mask.bool(),k=min_k)

        if self.is_oracle :
            
            leaf_indices = torch.nn.functional.pad(
                leaf_indices,
                pad=(0, self.n_schema_leafs - leaf_indices.size(-1)),
                mode="constant",
                value=-1,
            )
            leaf_schema_mask = (leaf_indices>=0)
            final_leaf_indices = torch.abs(leaf_indices)

        else:
            final_leaf_indices = top_agenda_indices

        leaf_schema_rep = allennlp.nn.util.batched_index_select(
            embedded_schema.contiguous(), final_leaf_indices
        )
        
        leaf_schema_hash = allennlp.nn.util.batched_index_select(leaf_hash.unsqueeze(-1),  final_leaf_indices).reshape([batch_size,-1])
        leaf_schema_types = allennlp.nn.util.batched_index_select(leaf_types.unsqueeze(-1), final_leaf_indices).reshape([batch_size,-1]).long()
        if self.value_pred:
            agenda_rep = torch.cat([leaf_schema_rep,leaf_span_rep],dim=-2)
            agenda_hash = torch.cat([leaf_schema_hash,leaf_span_hash],dim=-1)
            agenda_types = torch.cat([leaf_schema_types,leaf_span_types],dim=-1)
            agenda_mask = torch.cat([leaf_schema_mask,leaf_span_mask],dim=-1)
            item_list.append(ZeroItem(agenda_types,final_leaf_indices,span_start_indices,span_end_indices,orig_entities,enc,self.tokenizer))
        else:
            agenda_rep = leaf_schema_rep
            agenda_hash = leaf_schema_hash
            agenda_types = leaf_schema_types
            agenda_mask = leaf_schema_mask
            item_list.append(ZeroItem(agenda_types,final_leaf_indices,None,None,orig_entities,enc,self.tokenizer))

        
        
        
  
        
        outputs['leaf_agenda_hash'] = agenda_hash
        enc_list = [self.tokenizer.decode(enc["tokens"]['token_ids'][b].tolist()) for b in range(batch_size)]
        decoding_step = 0

        for decoding_step in range(self._decoder_timesteps):
            batch_size, seq_len, _ = agenda_rep.shape
            if self.cntx_beam:
                enriched_agenda_rep = self._augment_with_utterance(
                    embedded_utterance,
                    utterance_mask,
                    agenda_rep,
                    agenda_mask,
                    ctx=self._agenda_encoder,
                )
            else:
                enriched_agenda_rep = agenda_rep
            if self.cntx_rep:
                agenda_rep = enriched_agenda_rep.contiguous()
            # encoder_input = torch.cat([embedded_utterance, agenda_rep], dim=1)
            # agenda_sum = self.summrize_agenda(enriched_agenda_rep,agenda_mask.bool()).squeeze(1)
            # self.summrize_vec(0)
            # agenda_sum = self._pooler(enriched_agenda_rep,agenda_mask.bool())
            # depth_scores = self._score_depth(agenda_sum)
            # depth_loss += self.xent(depth_scores,depth_counter)

            
            frontier_scores, frontier_mask = self.score_frontier(enriched_agenda_rep,agenda_rep,agenda_mask)
            frontier_scores = frontier_scores/self.temperature
            l_agenda_idx, r_agenda_idx = compute_agenda_idx(batch_size,seq_len,self.binary_op_count,self.unary_op_count,device=self._device)
            frontier_op_ids = compute_op_idx(batch_size,seq_len,self.binary_op_count,self.unary_op_count,device=self._device)
            

            frontier_hash = self.hash_frontier(agenda_hash, frontier_op_ids, l_agenda_idx, r_agenda_idx)
            valid_op_mask = self.typecheck_frontier(agenda_types, frontier_op_ids, l_agenda_idx, r_agenda_idx)
            frontier_mask = frontier_mask * valid_op_mask

            unique_frontier_scores = frontier_scores

            if self.training:
                with torch.no_grad():
                    is_levelorder_list = frontier_utils.new_isin(new_hash_gold_levelorder[decoding_step + 1],frontier_hash)
                if self.use_bce:
                    bce_vector_loss += (self._bce_loss(frontier_scores,is_levelorder_list.float())*frontier_mask.bool()).mean()
                else:
                    avg_frontier_scores = util.masked_log_softmax(frontier_scores, frontier_mask.bool(), dim=1)
                    loss_tensor = -avg_frontier_scores * is_levelorder_list.float()
                    vector_loss += loss_tensor.squeeze().sum(-1)
                    tree_sizes_vector += is_levelorder_list.bool().squeeze().sum(-1)

                # so we won't pick more multiple items with the same encoding
                if self.uniquify:
                    is_levelorder_list = is_levelorder_list * unique_ids_list
                unique_frontier_scores = unique_frontier_scores.masked_fill(
                    is_levelorder_list.bool(), ai2_util.max_value_of_dtype(unique_frontier_scores.dtype)
                )

            agenda_scores, agenda_mask, agenda_idx = util.masked_topk(unique_frontier_scores,mask=frontier_mask.bool(),k=self._agenda_size)
            old_agenda_types = agenda_types.clone()
            
            agenda_types = torch.gather(frontier_op_ids,-1,agenda_idx)
            

            keep_indices = (agenda_types==self.keep_id).nonzero().t().split(1)
            l_child_idx = torch.gather(l_agenda_idx,-1,agenda_idx)
            r_child_idx = torch.gather(r_agenda_idx,-1,agenda_idx)
            child_types = util.batched_index_select(old_agenda_types.unsqueeze(-1),r_child_idx).squeeze(-1)
            
            agenda_rep = self._create_agenda_rep(agenda_rep, l_child_idx, r_child_idx, agenda_types, keep_indices)
            
            
            agenda_hash = torch.gather(frontier_hash,-1,agenda_idx)
            if decoding_step==1 and self.debug:
                failed_list,node_list,failed_set = get_failed_set(agenda_hash,decoding_step,tree_obj,batch_size,hash_gold_levelorder)
                if failed_set:
                    print("hi")

            item_list.append(Item(agenda_types,l_child_idx,r_child_idx,agenda_mask))
            agenda_types = torch.where(agenda_types==self.keep_id,child_types,agenda_types)
            agenda_list.append(
                [
                    agenda_rep.clone(),
                    agenda_hash.clone(),
                    agenda_mask.clone(),
                    agenda_types.clone(),
                    agenda_scores.clone(),
                ]
            )


        if self.should_rerank or not self.training:
            (
                enriched_agenda_rep_list,
                agenda_hash_list,
                agenda_mask_list,
                agenda_type_list,
                agenda_scores_list,
            ) = zip(*agenda_list)
            agenda_mask_tensor = torch.cat(agenda_mask_list, dim=1)
            agenda_type_tensor = torch.cat(agenda_type_list, dim=1)
            
            is_final_mask = (
                self._term_tensor[agenda_type_tensor].bool().to(agenda_mask_tensor.device)
            )
            agenda_mask_tensor = agenda_mask_tensor * is_final_mask
            agenda_hash_tensor = torch.cat(agenda_hash_list, dim=1)
            agenda_scores_tensor = torch.cat(agenda_scores_list, dim=1)
            agenda_scores_tensor = agenda_scores_tensor
            agenda_scores_tensor = agenda_scores_tensor.masked_fill(~agenda_mask_tensor.bool(),
                                    ai2_util.min_value_of_dtype(agenda_scores_tensor.dtype))
            enriched_agenda_rep_tensor = torch.cat(enriched_agenda_rep_list, dim=1)
        if self.should_rerank:
            
            if self.cntx_reranker:
                pass
            else:
                enriched_agenda_rep_tensor = self._noreranker_cntx_linear(enriched_agenda_rep_tensor)
            ranked_agenda = self._rank_agenda(enriched_agenda_rep_tensor).squeeze(-1)
            ranked_agenda = ranked_agenda.masked_fill(~agenda_mask_tensor.bool(), 
                                                        ai2_util.min_value_of_dtype(ranked_agenda.dtype))

            ranked_agenda = util.masked_softmax(
                ranked_agenda, agenda_mask_tensor.float(), dim=-1
            )
        else:
            ranked_agenda = None
        


        if self.training:
            pre_loss = (vector_loss / tree_sizes_vector).mean() 
            if self.use_bce:
                pre_loss += bce_vector_loss.mean()


            loss = pre_loss.squeeze()
            assert not bool(torch.isnan(loss))
            # loss += (depth_loss.squeeze()/steps)
            outputs["loss"] = loss
            self._compute_validation_outputs(
                outputs, hash_gold_tree, agenda_hash,
            )
            return outputs
        else:
            end = time.time()
            if tree_obj is not None:
                outputs['hash_gold_levelorder'] = [hash_gold_levelorder]
            self._compute_validation_outputs(
                outputs,
                hash_gold_tree,
                agenda_hash,
                is_gold_leaf=is_gold_leaf,
                top_agenda_indices=top_agenda_indices,
                db_id=db_id,
                agenda_hash_tensor=agenda_hash_tensor,
                agenda_scores_tensor=agenda_scores_tensor,
                agenda_scores=ranked_agenda,
                gold_sql=gold_sql,
                item_list=item_list,
                inf_time=end-start,
                total_time=end-total_start,
            )
            return outputs

    def score_spans(self, embedded_utterance,utterance_mask):
        batch_size,utterance_length,_ = embedded_utterance.shape

        logits = self._span_score_func(embedded_utterance)
        logits = logits/self.temperature
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_logits = frontier_utils.replace_masked_values_with_big_negative_number(start_logits,utterance_mask)
        end_logits = frontier_utils.replace_masked_values_with_big_negative_number(end_logits,utterance_mask)
        span_scores = frontier_utils.get_span_scores(start_logits,end_logits)
        return span_scores,start_logits,end_logits

    def summrize_agenda(self,enriched_agenda_rep,agenda_mask):
        indices = torch.zeros_like(agenda_mask[:,0]).long()
        sum_mask = torch.ones_like(agenda_mask[:,0]).bool().unsqueeze(-1)
        sum_vec = self.summrize_vec(indices).unsqueeze(1)
        input_mask = torch.cat([sum_mask, agenda_mask], dim=-1)
        encoder_input = torch.cat([sum_vec, enriched_agenda_rep], dim=1)
        encoder_output = self._agenda_summarizer(inputs=encoder_input, mask=input_mask)
        agenda_sum, enriched_agenda_rep = torch.split(
            encoder_output, [sum_mask.size(-1), agenda_mask.size(-1)], dim=1
        )
        return agenda_sum

    def _create_agenda_rep(self, agenda_rep, l_child_idx, r_child_idx, agenda_types, keep_indices):
        l_child_rep = util.batched_index_select(agenda_rep,l_child_idx)
        r_child_rep = util.batched_index_select(agenda_rep,r_child_idx)
        agenda_type_rep =  self.type_embedding(agenda_types)
        agenda_rep = torch.stack([agenda_type_rep,l_child_rep,r_child_rep],dim=-2)
        batch_size,agenda_size,_,emb_size = agenda_rep.shape
        agenda_rep = agenda_rep.reshape([-1,3,self._action_dim])
        mask = torch.ones([agenda_rep.size(0),3],dtype=torch.bool,device=self._device)
        agenda_rep = self._tree_rep_transformer(inputs=agenda_rep, mask=mask)
        agenda_rep = self._pooler(agenda_rep).reshape([batch_size,agenda_size,emb_size])
        
        agenda_rep[keep_indices] = r_child_rep[keep_indices].type(agenda_rep.dtype)
        return agenda_rep

    def _take_mml(self, loss_tensor, mask):
        s = loss_tensor.masked_fill(mask.bool(), 1)
        return -torch.log(s).sum(-1).mean()

    def _compute_validation_outputs(
        self,
        outputs,
        hash_gold_tree,
        agenda_hash,
        **kwargs,
    ):
        batch_size = agenda_hash.size(0)
        final_beam_acc_list = []
        reranker_acc_list = []
        spider_acc_list = []
        leaf_acc_list = []
        sql_list = []
        tree_list = []
        if hash_gold_tree is not None:
            for gs, fa in zip(hash_gold_tree, agenda_hash.tolist()):
                acc = int(gs) in fa
                self._final_beam_acc(int(acc))
                final_beam_acc_list.append(bool(acc))

        if not self.training:
            
            
            if kwargs['is_gold_leaf'] is not None  and kwargs['top_agenda_indices'] is not None:
                for top_agenda_indices_el, is_gold_leaf_el in zip(
                    kwargs["top_agenda_indices"], kwargs["is_gold_leaf"]
                ):
                    is_gold_leaf_idx = is_gold_leaf_el.nonzero().squeeze().tolist()
                    leaf_acc = int(
                        all([x in top_agenda_indices_el for x in is_gold_leaf_idx])
                    )
                    leaf_acc_list.append(leaf_acc)
                    self._leafs_acc(leaf_acc)

            #TODO: change this!! this causes bugs!
            for b in range(batch_size):
                if self.should_rerank:
                    agenda_scores_el = kwargs["agenda_scores"][b]
                else:
                    agenda_scores_el = kwargs["agenda_scores_tensor"][b]
                    agenda_scores_el[:-self._agenda_size]= ai2_util.min_value_of_dtype(agenda_scores_el.dtype)
                top_idx = int(agenda_scores_el.argmax())
                tree_copy = ''
                try:
                    items = kwargs["item_list"][:(top_idx//self._agenda_size)+2]
                    
                    tree_res = node_util.reconstruct_tree(self._op_names,self.binary_op_count, b,top_idx%self._agenda_size,items,len(items)-1,self.n_schema_leafs)

                    tree_res = node_util.remove_keep(tree_res)
                    # tree_res = self.replacer.post(tree)
                    tree_copy = deepcopy(tree_res)
                    sql = node_util.print_sql(tree_res)
                    sql = node_util.fix_between(sql)
                    sql = sql.replace("LIMIT value","LIMIT 1")
                    # TOM this is what we get eventually and should be the same.

                    
                except:
                    print("damn")
                    sql = ''
                spider_acc = 0
                reranker_acc = 0
                
                outputs["inf_time"] = [kwargs['inf_time']]
                outputs["total_time"] = [kwargs['total_time']]

                if hash_gold_tree is not None:
                    # try:
                    reranker_acc = int(
                        kwargs["agenda_hash_tensor"][b][top_idx]== int(hash_gold_tree[b])
                    )

                    gold_sql = kwargs["gold_sql"][b]
                    db_id = kwargs["db_id"][b]
                    spider_acc = int(
                        self._evaluate_func(
                            gold_sql, sql, db_id
                        )
                    )
                    # except:
                    #     print("fuck")

                reranker_acc_list.append(reranker_acc)
                self._reranker_acc(reranker_acc)
                self._spider_acc(spider_acc)
                # print(self._spider_acc._count)
                sql_list.append(sql)
                tree_list.append(tree_copy)
                spider_acc_list.append(spider_acc)
            outputs["agenda_scores"] = [agenda_scores_el]
            outputs["agenda_encoding"] = [kwargs["item_list"]]
            outputs["agenda_hash"] = [kwargs["agenda_hash_tensor"]]
            if hash_gold_tree is not None:
                outputs["gold_hash"] = hash_gold_tree
            outputs["reranker_acc"] = reranker_acc_list
            outputs["spider_acc"] = spider_acc_list
            outputs['sql_list'] = sql_list
            outputs['tree_list'] = tree_list
        outputs["final_beam_acc"] = final_beam_acc_list
        outputs["leaf_acc"] = leaf_acc_list
        
    def _augment_with_utterance(
        self, embedded_utterance, utterance_mask, agenda_rep, agenda_mask, ctx=None,
    ):
        assert ctx
        
        
        if self.disentangle_cntx:
            enriched_agenda_rep = self._utterance_augmenter(agenda_rep, embedded_utterance, ctx_att_mask=utterance_mask)[0]
            if self.cntx_agenda:
                enriched_agenda_rep = ctx(inputs=enriched_agenda_rep, mask=agenda_mask.bool())
        else:
            encoder_input = torch.cat([embedded_utterance, agenda_rep], dim=1)
            input_mask = torch.cat([utterance_mask.bool(), agenda_mask.bool()], dim=-1)
            encoder_output = ctx(inputs=encoder_input, mask=input_mask)
            _, enriched_agenda_rep = torch.split(
                encoder_output, [utterance_mask.size(-1), agenda_mask.size(-1)], dim=1
            )


             
        return enriched_agenda_rep



    def emb_q(self,enc):
        pad_dim = enc['tokens']['mask'].size(-1)
        if pad_dim>512:
            for key in enc['tokens'].keys():
                enc['tokens'][key] = enc['tokens'][key][:,:512]

            embedded_utterance_schema = self._question_embedder(enc)
        else:
            embedded_utterance_schema = self._question_embedder(enc)

        return embedded_utterance_schema

    def _encode_utt_schema(self, enc, offsets, relation, lengths):
        # with torch.no_grad():
        embedded_utterance_schema = self.emb_q(enc)

        (
            embedded_utterance_schema,
            embedded_utterance_schema_mask,
        ) = batched_span_select(embedded_utterance_schema, offsets)
        embedded_utterance_schema = masked_mean(
            embedded_utterance_schema,
            embedded_utterance_schema_mask.unsqueeze(-1),
            dim=-2,
        )
        # if self.training:
            # max_norm = torch.linalg.norm(embedded_utterance_schema,dim=-1).max().item()
            # self.max_norm = max(self.max_norm,max_norm)
        relation_mask = (relation >= 0).float()  # TODO: fixme
        torch.abs(relation, out=relation)
        embedded_utterance_schema = self._emb_to_action_dim(embedded_utterance_schema)
        enriched_utterance_schema = self._schema_encoder(
            embedded_utterance_schema, relation.long(), relation_mask
        )

        utterance_schema, utterance_schema_mask = batched_span_select(
            enriched_utterance_schema, lengths
        )
        utterance, schema = torch.split(utterance_schema, 1, dim=1)
        utterance_mask, schema_mask = torch.split(utterance_schema_mask, 1, dim=1)
        utterance_mask = torch.squeeze(utterance_mask, 1)
        schema_mask = torch.squeeze(schema_mask, 1)
        embedded_utterance = torch.squeeze(utterance, 1)
        # if not self.disentangle_cntx:
            # embedded_utterance = self._emb_to_action_dim(embedded_utterance)
        schema = torch.squeeze(schema, 1)
        # embedded_utterance = util.add_positional_features(embedded_utterance)
        return schema, schema_mask, embedded_utterance, utterance_mask


    def score_frontier(self, enriched_agenda_rep,agenda_rep, agenda_mask):
        if self.cntx_rep:
            agenda_rep = self._cntx_rep_linear(enriched_agenda_rep)
        else:
            if self.cntx_beam:
                agenda_rep = torch.cat([enriched_agenda_rep,agenda_rep],dim=-1)
                if self.lin_after_cntx:
                    agenda_rep = self.cntx_linear(agenda_rep)
            else:
                agenda_rep = self._nobeam_cntx_linear(agenda_rep)
    
        batch_size,seq_len,emb_size = agenda_rep.shape
        if not self.use_add:
            binary_ops_reps = (
                self._binary_frontier_embedder(agenda_rep, agenda_rep)
                .reshape(-1, self.d_frontier, seq_len ** 2)
                .transpose(-1, -2)
            )

        else:
            left = self.left_emb(agenda_rep.reshape([batch_size,seq_len,1,emb_size]))
            right = self.right_emb(agenda_rep.reshape([batch_size,1,seq_len,emb_size]))
            binary_ops_reps = self.after_add(left+right)
            binary_ops_reps = (binary_ops_reps.reshape(-1, seq_len ** 2,self.d_frontier)
                
            )
        unary_ops_reps = self._unary_frontier_embedder(agenda_rep)
        pre_frontier_rep = torch.cat([binary_ops_reps, unary_ops_reps], dim=1)
        pre_frontier_rep =  self.pre_op_linear(pre_frontier_rep)

        base_frontier_scores = self.op_linear(pre_frontier_rep)
        binary_frontier_scores, unary_frontier_scores = torch.split(
            base_frontier_scores, [seq_len ** 2, seq_len], dim=1
        )
        binary_frontier_scores, _ = torch.split(
            binary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        )
        _, unary_frontier_scores = torch.split(
            unary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        )
        frontier_scores = torch.cat(
            [
                binary_frontier_scores.reshape([batch_size, -1]),
                unary_frontier_scores.reshape([batch_size, -1]),
            ],
            dim=-1,
        )
        binary_mask = torch.einsum("bi,bj->bij", agenda_mask, agenda_mask)
        binary_mask = binary_mask.view([agenda_mask.shape[0], -1]).unsqueeze(-1)
        binary_mask = binary_mask.expand([batch_size, seq_len ** 2, self.binary_op_count]).reshape(batch_size, -1)
        unary_mask = (
            agenda_mask.clone()
            .unsqueeze(-1)
            .expand([batch_size, seq_len, self.unary_op_count])
            .reshape(batch_size, -1)
        )
        frontier_mask = torch.cat([binary_mask, unary_mask], dim=-1)
        
        return frontier_scores, frontier_mask

    # TODO: fix keep op
    # @classmethod
    def hash_frontier(self, agenda_hash, frontier_op_ids, l_agenda_idx, r_agenda_idx):
        # if 
        # if len(agenda_hash.shape)==1:
        #     agenda_hash = agenda_hash.unsqueeze(0)
        r_hash = (
            util.batched_index_select(agenda_hash.unsqueeze(-1), r_agenda_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_hash = (
            util.batched_index_select(agenda_hash.unsqueeze(-1), l_agenda_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        tmp = l_hash.clone()
        frontier_hash = self.set_hash(
            frontier_op_ids.clone().reshape(-1), l_hash, r_hash
        ).long()
        frontier_hash = torch.where(frontier_op_ids.reshape(-1)==self.keep_id, tmp, frontier_hash)
        frontier_hash = frontier_hash.reshape(r_agenda_idx.size())
        return frontier_hash

    
    def typecheck_frontier(self, agenda_types, frontier_op_ids, l_agenda_idx, r_agenda_idx):
        batch_size,frontier_size = frontier_op_ids.shape
        # if len(agenda_types.shape)==1:
        #     agenda_types = agenda_types.unsqueeze(0)
        r_types = (
            util.batched_index_select(agenda_types.unsqueeze(-1), r_agenda_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_types = (
            util.batched_index_select(agenda_types.unsqueeze(-1), l_agenda_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        indices_into = self._op_count*self._op_count*frontier_op_ids.view(-1) +self._op_count*l_types+r_types
        valid_ops = self._rule_tensor_flat[indices_into].reshape([batch_size,frontier_size])
        return valid_ops
    

    
    def set_hash(self, parent, a, b):
        a <<= 28
        b >>= 1
        a = a.add_(b)
        parent <<= 56
        a = a.add_(parent)
        a *= self.hasher.tensor2
        #TODO check lgu-lgm hashing instead of this:
        a = a.fmod_(self.hasher.tensor1)
        return a


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        out = {
            "final_beam_acc": self._final_beam_acc.get_metric(reset),
        }
        if not self.training:
            out["spider"] = self._spider_acc.get_metric(reset)
            out["reranker"] = self._reranker_acc.get_metric(reset)
            out["leafs_acc"] = self._leafs_acc.get_metric(reset)
            # out['self._spider_acc._count'] = self._spider_acc._count
        return out