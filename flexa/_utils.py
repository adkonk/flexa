import numpy as np
from scipy.stats import rankdata

def sortkey(i, j=None): 
    """Returns sorted tuple (min, max) for tuple i or values i, j"""
    if isinstance(i, tuple):
        return(min(i), max(i))
    assert i != j, '%d, %d' % (i, j)
    return(min(i, j), max(i, j))

def pairs(vals):
    """Returns cyclic pairs (vals[i], vals[i+1]) in a list"""
    return([(vals[i], vals[(i + 1) % len(vals)]) for i in range(len(vals))])

def reindex_list_of_lists(data, keep=None, default=-1):
    """Replaces subelements in a list of lists with a default value and 
    reindexes the elements

    Args:
        data (list of lists): to be reindexed
        keep (array_like, optional): elements of data to be kept. 
            Defaults to None.
        default (int, optional): value to replace elements not in keep with. 
            Defaults to -1.
        
    Returns:
        list of lists in the shape of data
    """
    # remove elements not in keep
    if keep is not None: 
        for (si, dset) in enumerate(data):
            for i in range(len(dset)):
                if dset[i] not in keep:
                    dset[i] = default
            data[si] = dset 

    # reindex
    i = 0
    flat = [d for dset in data for d in dset]
    flat = rankdata(flat, method='dense') - 1 + default
    # reshape to original sublist lengths
    for (si, dset) in enumerate(data):
        data[si] = list(flat[i:(i + len(dset))])
        i += len(dset)
    return(data)