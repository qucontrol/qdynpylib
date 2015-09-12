r'''
Decorator for function memoization

Memoization (sic!) is the concept caching the result of a function, so that if
the function is called again with the same parameters, the result is returned
directly, without recalculating it.

>>> from click import echo
>>> @memoize
... def f(a,b,c):
...     return str(a) + str(b) + str(c)
...
>>> echo(f(1,2,3))
123
>>> echo(f([], None, (1,2,3)))
[]None(1, 2, 3)
>>> len(f._combined_cache())
2

The memoize decorator allows for the possibility to write the cache to disk and
to reload it at a later time

>>> f.dump('memoize.dump')
>>> f.clear()
>>> len(f._combined_cache())
0
>>> f.load('memoize.dump')
>>> len(f._combined_cache())
2
>>> import os
>>> os.unlink('memoize.dump')
'''
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
from six.moves import cPickle as pickle
from functools import update_wrapper
from QDYN.io import open_file


class memoize(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''

    def clear(self):
        """Forget all cached results"""
        self.cache = {}
        self.pickle_cache = {}

    def __init__(self, func):
        """Create a memoized version of the given function"""
        self.cache = {}
        self.pickle_cache = {}
        self.func = func
        self.autosave = 0
        self._count = 0
        self.cache_file = None
        update_wrapper(self, func)

    def dump(self, filename):
        """Write memoization cache to the given file or file-like object"""
        with open_file(filename, 'wb') as pickle_file:
            pickle.dump((self.cache, self.pickle_cache), pickle_file)

    def load(self, filename, raise_error=False):
        """
        Load dumped data from the given file and update the
        memoization cache

        If raise_error is True, raise an IOError if the file does not exist.
        Otherwise, simply leave cache unchanged.
        """
        try:
            with open_file(filename, 'rb') as pickle_file:
                cache, pickle_cache = pickle.load(pickle_file)
                self.cache.update(cache)
                self.pickle_cache.update(pickle_cache)
        except IOError:
            if raise_error:
                raise

    def __call__(self, *args, **kwargs):
        if not kwargs:
            key = args
            try:
                return self.cache[key]
            except KeyError:
                value = self.func(*args)
                self.cache[key] = value
                return value
            except TypeError:
                # unhashable -- for instance, passing a list or dict as an
                # argument.  fall through to using pickle
                pass
        try:
            key = (pickle.dumps(args, 1), pickle.dumps(kwargs, 1))
        except TypeError:
            # probably have a function being passed in
            return self.func(*args, **kwargs)

        try:
            return self.pickle_cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.pickle_cache[key] = value
            return value

    def add_to_cache(self, result, args=None, kwargs=None):
        """Add the known result for a given set of args and kwargs to the
        cache. One case where this is useful is when calling a function as part
        of a multiprocessing pool, or another parallelization method that does
        not have shared memory. In this situation, memoization will fail, but
        the results may be added to the cache later on."""
        if not kwargs is None:
            try:
                self.cache[args] = result
                return
            except TypeError:
                # unhashable -- for instance, passing a list or dict as an
                # argument.  fall through to using pickle
                pass
        key = (pickle.dumps(args, 1), pickle.dumps(kwargs, 1))
        self.pickle_cache[key] = result

    def _combined_cache(self):
        """Return the combined cache and pickle_cache"""
        result = {}
        result.update(self.cache)
        result.update(self.pickle_cache)
        return result

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
