"""tiff_pickle_test.py
Times how quick tiffs and pickled Tiff objects are loaded.
Run with `python tiff_pickle_test.py`
"""

from functools import wraps
from time import time
from pathlib import Path
import pickle
import napari
from gulp2p.preproc.tiff import Tiff

TIFF_PATH = Path(r"Z:\2PImaging\Kerstin\MIMS\20231207\20231207_DR019xiGluSnFr_Fly1_00002.tif")
PICKLE_PATH = Path(r"Z:\test_tiff_pickle")

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #   (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        
        return result
    return wrap

@timing
def load_tiff():
    return Tiff(TIFF_PATH)

@timing
def dump_pickle(tiff):
    with open(PICKLE_PATH, 'rb+') as outfile:
        pickle.dump(tiff, outfile)

@timing
def create_viewer():
    return napari.Viewer()

@timing
def load_pickled_tiff():
    with open(PICKLE_PATH, 'rb') as outfile:
        loaded_tiff = pickle.load(outfile)
    return loaded_tiff

@timing
def get_metadata(tiff):
    return tiff.metadata 

@timing
def get_stack(tiff):
    return tiff.stack 

@timing
def open_image(viewer, stack):
    viewer.add_image(stack)


@timing
def main():
    print("Create tiff pickle:")
    viewer = create_viewer()
    tiff = load_tiff()
    dump_pickle(tiff)
    stack = get_stack(tiff)
    print("Create tiff pickle again:")
    # I think the second time it loads faster since it is in ram or some cache that the os handles.
    viewer = create_viewer()
    tiff = load_tiff()
    dump_pickle(tiff)
    stack = get_stack(tiff)
    print("load pickled tiff:")
    loaded_tiff = load_pickled_tiff()
    stack = get_stack(loaded_tiff)
    stack = get_stack(loaded_tiff)
    open_image(viewer, stack)
    print("load pickled tiff again:")
    # I think the second time it loads faster since it is in ram or some cache.
    loaded_tiff = load_pickled_tiff()
    stack = get_stack(loaded_tiff)
    stack = get_stack(loaded_tiff)

if __name__ == "__main__":
    main()
