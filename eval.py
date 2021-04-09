import os
import argparse        
import torch

from allennlp.models.archival import Archive,load_archive,archive_model
import pathlib
import random
from allennlp.data.vocabulary import Vocabulary
from modules.relation_transformer import *
from training.callbacks import *
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_path', type=str, default="/home/ohadr/experiments/paltry-crimson-corgi_gpu7")
    parser.add_argument('--dev_path', type=str, default='dataset/dev.json')
    parser.add_argument('--table_path', type=str, default='dataset/tables.json')
    parser.add_argument('--dataset_path', type=str, default='dataset/database')
    parser.add_argument('--output', type=str,default="predictions_with_vals_fixed4.txt")
    parser.add_argument('--gpu', type=int,default=0)
    args = parser.parse_args()
    
    overrides = {"dataset_reader":{"tables_file":args.table_path,"dataset_path":args.dataset_path}}
    overrides["validation_dataset_reader"] = {"tables_file":args.table_path,"dataset_path":args.dataset_path}
    predictor = Predictor.from_path(args.archive_path,cuda_device=args.gpu,overrides=overrides)
    print("after pred")

    with open(args.output,"w") as g:
        with open(args.dev_path) as f:
            dev_json = json.load(f)
            for i,el in enumerate(tqdm.tqdm(dev_json)):
                instance = predictor._dataset_reader.text_to_instance(utterance=el['question'],db_id=el['db_id'])
                if i==0:
                    instance_0 = instance
                if instance is not None:
                    with torch.cuda.amp.autocast(enabled=True):
                        out = predictor._model.forward_on_instances([instance,instance_0])
                        pred = out[0]['sql_list']
                        
                else:
                    pred = 'NO PREDICTION'
                g.write(f"{pred}\t{el['db_id']}\n")    
if __name__=='__main__':
    main()
