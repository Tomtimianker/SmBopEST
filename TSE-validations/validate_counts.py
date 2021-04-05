import dill
import json
from dataset_readers.sparc_reader import SparcNaiveDatasetReader

TRAIN = r'dataset/sparc/train.json'
TRAIN_CACHE = r'/specific/netapp5/joberant/home/ohadr/smbop/edan/SmBopEST/cache/exp200train/dataset_SLASH_sparc_SLASH_train.json'

DEV = r'dataset/sparc/dev.json'
DEV_CACHE = r'/specific/netapp5/joberant/home/ohadr/smbop/edan/SmBopEST/cache/exp200val/dataset_SLASH_sparc_SLASH_dev.json'

## count how many examples exist in the original train-set:
with open(TRAIN) as f:
    j = json.load(f)
    orig_train = sum([1 for item in SparcNaiveDatasetReader.enumerate_json(None, j)])
    train_interactions = len(j)

with open(TRAIN_CACHE, 'rb') as f:
    d = dill.load(f)
    parsed_train = len(d)

with open(DEV) as f:
    j = json.load(f)
    orig_dev = sum([1 for item in SparcNaiveDatasetReader.enumerate_json(None, j)])
    dev_interactions = len(j)

with open(DEV_CACHE, 'rb') as f:
    d = dill.load(f)
    parsed_dev = len(d)

print(f'Original dataset contains: {orig_train} trining examples, from {train_interactions} interactions, of which {parsed_train} were succesfully parsed ({"{:.3%}".format(parsed_train * 1.0 / orig_train)})')
print(f'Original dataset contains: {orig_dev} dev examples,  from {dev_interactions} interactions, of which {parsed_dev} were succesfully parsed ({"{:.3%}".format(parsed_dev * 1.0 / orig_dev)})')



