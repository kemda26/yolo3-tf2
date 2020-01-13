import requests
import os
from tqdm import tqdm
import numpy as np
import glob, shutil, csv


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


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        path = os.path.join('log-files', logname, now)
        os.makedirs(path)
        folders = [('','*.py'), ('yolo/net/','*.py'), ('configs/','*')]
        filenames = glob.glob('*.py') + glob.glob('yolo/net/*.py') + glob.glob('configs/*')  # put copy of all python files in log_dir
        for folder, query in folders:
            filenames = glob.glob(folder + query)
            path_ = os.path.join(path, folder)
            if not os.path.exists(path_): os.makedirs(path_)
            for filename in filenames:     # for reference
                shutil.copy(filename, path_)
        imgfile = os.path.join(path, 'img.csv')
        logfile = os.path.join(path, 'log.csv')

        self.write_header = True
        self.write_header_img = True
        # self.log_entry = {}
        self.f = open(logfile, 'w')
        self.img_f = open(imgfile, 'w')
        self.writer = None  # DictWriter created with first call to write() method
        self.writer_img = None

    def write_img(self, input):
        if self.write_header_img:
            fieldnames = [x for x in input.keys()]
            self.writer_img = csv.DictWriter(self.img_f, fieldnames=fieldnames)
            self.writer_img.writeheader()
            self.write_header_img = False
        self.writer_img.writerow(input)        

    def write(self, input, display=True) -> 'input is a dictionary':
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(input)
        if self.write_header:
            fieldnames = [x for x in input.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(input)
        # self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        # print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                            #    log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()