#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime

def timer(func):
    '''This function shows the execution time of the function object passed'''
    def wrap_func(*args, **kwargs):
        start = datetime.datetime.now()
        try:
            result = func(*args, **kwargs)
            stop = datetime.datetime.now()
            delta = str(stop-start).replace(':','h',1).replace(':','m',1) + 's'
            print(f'{func.__name__!r}: successfully processed in {delta}.')
#           print(f'{func.__name__!r} done in {(stop-start):.4f}s')
            return result
        except KeyboardInterrupt:
            stop = datetime.datetime.now()
            delta = str(stop-start).replace(':','h',1).replace(':','m',1) + 's'
            print(f'{func.__name__!r}: manually interrupted after a {delta} run.')
    return wrap_func
