import numpy as np
from scipy.stats import rankdata
from collections.abc import Iterable

def sortkey(i, j=None): 
    """Returns sorted tuple (min, max) for tuple i or values i, j"""
    if isinstance(i, Iterable):
        return(min(i), max(i))
    assert i != j, '%d, %d' % (i, j)
    return(min(i, j), max(i, j))

def pairs(vals):
    """Returns cyclic pairs (vals[i], vals[i+1]) in a list"""
    if len(vals) <= 2:
        raise ValueError('Must be at least 3 points to traverse in a cycle')
    return([(vals[i], vals[(i + 1) % len(vals)]) for i in range(len(vals))])

def rowwise_dot(a, b, **kwargs):
    """Returns rowwise dot product between matrices a, b"""
    return np.sum(a * b, axis=1, **kwargs)

def reindex_list_of_lists(lol, keep=None):
    """Replaces reindexes all the subelements in a list of lists by removing 
    anything not in list keep. Anything not in keep is replaced with -1

    Args:
        lol (list of lists): to be reindexed
        keep (array_like, optional): elements of data to be kept. 
            Defaults to None.
        
    Returns:
        list of lists in the shape of data
    """
    data = lol.copy() # function modifies list outside its environment otherwise
    # remove elements not in keep
    if keep is not None: 
        for (si, dset) in enumerate(data):
            for i in range(len(dset)):
                if dset[i] not in keep:
                    dset[i] = -1
            data[si] = dset 
        offset = 0
    else:
        offset = 1

    # reindex
    i = 0
    flat = [d for dset in data for d in dset]
    flat = rankdata(flat, method='dense') - 2 + offset
    # reshape to original sublist lengths
    for (si, dset) in enumerate(data):
        data[si] = list(flat[i:(i + len(dset))])
        i += len(dset)
    return(data)

def picklify(name):
    """Adds '.p' to file names if not already present"""
    ext = '.p'
    if not name.lower().endswith(ext):
        name = name + ext
    return(name)