import dill

def highest_in_cache_file(cache_filename: str):
    hights = [0] * 20
    with open(cache_filename, "rb") as cache_file:
        d = dill.load(cache_file)
        for x in d:
            if "depth" not in x.fields:
                hights[0] += 1
            else:
                hights[x.fields["depth"].array.shape[0]] += 1
    print(hights)


highest_in_cache_file('/a/home/cc/students/cs/ohadr/netapp/smbop/edan/SmBopEST/cache/exp200train/dataset_SLASH_sparc_SLASH_train.json')
