import requests
import os
from tqdm import tqdm
import numpy as np

def download_file(filename, url):
    print("Download {} from {}".format(filename, url))
    
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename


def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        download_file(filename, url)
        return True
    return False


class FileSorter:
    def __init__(self):
        pass
    
    def sort(self, list_of_strs):
        list_of_strs.sort(key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s
    
    def _alphanum_key(self, s):
        import re
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]


class EarlyStopping(object):
    def __init__(self, mode='min', delta=0, patience=10, percentage=False):
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False


    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False


    def _init_is_better(self, percentage):
        if self.mode not in ['min', 'max']:
            raise ValueError('mode ' + self.mode + ' is unknown!')
        if not percentage:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - self.delta
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + self.delta
        else:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - (best * self.delta / 100)
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + (best * self.delta / 100)
