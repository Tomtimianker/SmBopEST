import dill
import matplotlib.pyplot as plt
import numpy as np

total = []

with open(r'/a/home/cc/students/cs/ohadr/netapp/smbop/edan/SmBopEST/cache/exp200train/dataset_SLASH_sparc_SLASH_train.json', "rb") as cache_file:
    d = dill.load(cache_file)
    for instance in d:
        enc = instance['enc']
        total.append(len(enc.tokens))
total = np.array(total)
_ = plt.hist(total, bins='auto')
plt.savefig('hist')

print(np.percentile(total, 90))