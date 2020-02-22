#!/usr/bin/python
# -*- coding: utf-8 -*-

# python imports
import os
import sys
import logging
import logging.handlers
import platform
import time
from pympler import asizeof
from multiprocessing import Process

from time import sleep


import sys
from numbers import Number
from collections import Set, Mapping, deque

try: # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

logger = logging.getLogger(__file__)


def get_home_dir():
    """Return the user home directory"""
    return os.path.expanduser('~')


def get_app_home(app_name):
    """Return the application home directory. This will be a directory
    in $HOME/.app_name/ but with app_name lower cased.
    """
    return os.path.join(get_home_dir(), '.' + app_name.lower())


def get_root_dir():
    """Return the root directory of the application"""

    # was run from an executable?
    root_d = os.path.dirname(sys.argv[0])
    if not root_d:
        root_d = get_module_path()

    return root_d


def get_module_path():
    '''Get the path to the current module no matter how it's run.'''
    if '__file__' in globals():
        # If run from py
        return os.path.dirname(__file__)

    # If run from command line or an executable
    return get_root_dir()


def get_module_pkg():
    """Return the module's package path.
    Ej:
        if current module is imagis.utils.common the the call to:
            get_module_pkg()
        should return: imagis.utils
    """
    return '.'.join(__name__.split('.')[:-1])


def get_all_filenames(root_path=None):
    root_path = get_module_path() if root_path is None else root_path
    files_list = []

    for path, dir, files in os.walk(root_path):

        for _file in files:
            if len(dir) == 1:
                str_to_print = "{0}/{1}/{2}".format(path, dir, _file)

            else:
                str_to_print = "{0}/{1}".format(path, _file)

            if str_to_print.find("hyper_params")  != -1:
                files_list.append(str_to_print)

    return files_list

class ExecutionTime(object):
    """
    Helper that can be used in with statements to have a simple
    measure of the timming of a particular block of code, e.g.
    with ExecutionTime("db flush"):
        db.flush()
    """
    def __init__(self, info="", with_traceback=False):
        self.info = info
        self.with_traceback = with_traceback

    def __enter__(self):
        self.now = time.time()

    def __exit__(self, type, value, stack):
        logger = logging.getLogger(__file__)
        msg = '%s: %s' % (self.info, time.time() - self.now)
        if logger.handlers:
            logger.debug(msg)
        else:
            print(msg)
        if self.with_traceback:
            import traceback
            msg = traceback.format_exc()
            if logger.handlers:
                logger.error(msg)
            print (msg)


def get_obj_size(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)
    # return get_obj_size2(obj_0)


def get_obj_size2(obj):
    return asizeof.asizeof(obj)



# En multiprocesamiento, los procesos se generan creando un objeto Process

# y luego llamando a su método start()

# Process utiliza la API de threading.Thread


def exec_with_timeout(func, args, time):
    """

    Ejecuta una función con un limite de tiempo

    Tiene que recibir:

        func: el nombre de la función a ejecutar

        args: una tupla con los argumentos a pasar a la función

    Devuelve True si ha finalizado la función correctamente


    https://docs.python.org/2/library/multiprocessing.html

    """

    p = Process(target=func, args=args)

    p.start()

    p.join(time)

    if p.is_alive():
        p.terminate()

        print("Ha finalizado por timeout")

        return False

    print("Se ha ejecutado correctamente")

    return True

