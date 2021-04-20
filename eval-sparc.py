import os
import argparse        
import torch

from allennlp.models.archival import Archive,load_archive,archive_model
import pathlib
import random
from allennlp.data.vocabulary import Vocabulary
from modules.relation_transformer import *
# from training.callbacks import *
import json
from allennlp.common import Params
from models.semantic_parsing.smbop import SmbopParser
from modules.lxmert import LxmertCrossAttentionLayer
from dataset_readers.smbop import SmbopDatasetReader
import itertools
import utils.node_util as node_util
import numpy as np
from functools import partial
import numpy as np
from dataclasses import dataclass
import json
from functools import partial
import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import re
import json
import dill
from collections import defaultdict

from dataset_readers.sparc_reader import SparcNaiveDatasetReader
from dataset_readers.dev_sparc_reader import SparcDevNaiveDatasetReader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_path', type=str, default="/mnt/netapp7/ohadr/edanzwick/winning/SmBopEST/experiments/shaky-viridian-manatee_gpu7_batch_size5")
    parser.add_argument('--dev_path', type=str, default='dataset/sparc/dev.json')
    parser.add_argument('--table_path', type=str, default='dataset/sparc/tables.json')
    parser.add_argument('--dataset_path', type=str, default='dataset/sparc/database')
    parser.add_argument('--cache_file', type=str, default='/mnt/netapp7/ohadr/edanzwick/winning/SmBopEST/cache/exp200val/dataset_SLASH_sparc_SLASH_dev.json')
    parser.add_argument('--output', type=str,default="predictions_with_vals.txt")
    parser.add_argument('--gpu', type=int,default=3)
    args = parser.parse_args()
    
    print("reading predictor")
    overrides = {"dataset_reader":{"tables_file":args.table_path,"dataset_path":args.dataset_path}}
    overrides["validation_dataset_reader"] = {"tables_file":args.table_path,"dataset_path":args.dataset_path}
    predictor = Predictor.from_path(args.archive_path,cuda_device=args.gpu,overrides=overrides)
    print("after pred")

    cache = defaultdict(lambda: dict())

    with open(args.cache_file, "rb") as f:
        dev = dill.load(f)
        for i,instance in enumerate(tqdm.tqdm(dev)):
            if i==0:
                instance_0 = instance
            if instance is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    out = predictor._model.forward_on_instances([instance,instance_0])
                    pred = out[0]['sql_list']
            else:
                pred = 'NO PREDICTION'
            cache[instance.fields['major'].label][instance.fields['minor'].label] = f"{pred}\t{instance['db_id'].metadata}\n"

    # passed examples to the model in a wierd order so all dependancies will be met when an example is read.
    # Now we need to write them to file in order.
    with open(args.output,"w") as g:
        with open(args.dev_path) as d:
            j = json.load(d)
            prev_major = 0
            for i, example in SparcNaiveDatasetReader.enumerate_json(None, j):
                major = example['major']
                minor = example['minor']
                if major != prev_major:
                    g.write('\n')
                    prev_major = major
                if major not in cache or minor not in cache[major]:
                    g.write('NO PREDICTION\n')
                    continue
                g.write(cache[major][minor])

if __name__=='__main__':
    main()
