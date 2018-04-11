import random
import numpy as np
import time
import multiprocessing
from multiprocessing.managers import BaseManager

LEN = 1000000 * 100 * 4
OVERRIDE_CC = None

def make_col(val):
    return val * np.arange(LEN)

class MyManager(BaseManager): pass

def Manager():
    m = MyManager()
    m.start()
    return m 

class Counter(object):
    def __init__(self):
        self.ref_body = {
            'a': make_col(1),
            'b': make_col(2),
            'c': make_col(3)}

    def update(self, value):
        rb = self.ref_body
        
        for key in rb.keys():
            # perform some computation
            rb[key].mean() * value

MyManager.register('Counter', Counter)

def update(counter_proxy, thread_id):
    time.sleep(random.random() * 0.5)
    counter_proxy.update(1)
    print('on t_id %s' % thread_id)
    return counter_proxy

def main():
    manager = Manager()
    counter = manager.Counter()
    cc = multiprocessing.cpu_count()

    if OVERRIDE_CC:
        cc = OVERRIDE_CC
    print('cpus', cc)

    pool = multiprocessing.Pool(cc)
    pool.starmap(update, [(counter, i) for i in range(10)])

    print('done')