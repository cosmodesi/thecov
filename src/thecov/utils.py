"""This module contains utility functions for thecov.
"""
import os, functools

def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return
    
# python's enumerate but with a custom step = 2
def enum2(xs, start=0, step=2):
    """Enumerate a sequence with a custom step.

    Parameters
    ----------
    xs : sequence
        Sequence to enumerate.
    start : int, optional
        Starting index. Default is 0.
    step : int, optional
        Step of the enumeration. Default is 2.

    Returns
    -------
    generator
        Generator of tuples (index, element).
    """
    for x in xs:
        yield (start, x)
        start += step

def limit(iterable, count):
    """
    Limit number of iterated elements from an iterable.
    count -- is the maximum number of elements to iterate through
    """
    while count > 0:
        yield next(iterable)
        count -= 1

def cache_method(func):
    '''Decorator to cache the result of a method.

    Parameters
    ----------
    func : callable
        Method to cache.

    Returns
    -------
    callable
        Cached method.
    '''

    @functools.wraps(func)
    def cached_func(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = {}
        
        if len(args) + len(kwargs) == 1:
            key = args[0] if args else next(iter(kwargs.values()))
        else:
            key = hash((args, frozenset(kwargs.items())))

        cache = self._cache[func.__name__]
        
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
    
        return cache[key]

    return cached_func