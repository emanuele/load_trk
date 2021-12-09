import numpy as np
from load_trk import load_streamlines
from load_trk_new import load_streamlines as load_streamlines_new
from load_trk_numba import load_streamlines as load_streamlines_numba
from time import time

if __name__=='__main__':

    n_streamlines = int(4e6) # Use None if you want to read all streamlines
    
    filename = 'data/sub-599469_var-10M_tract.trk'

    print("Using load_trk.py")
    t0 = time()
    streamlines, header, lengths, idxs = load_streamlines(filename, idxs=range(n_streamlines), apply_affine=False, container='array', verbose=True)
    print(f"Total time: {time() - t0} sec.")

    print("")
    print("Using load_trk_new.py")
    t0 = time()
    streamlines, header, lengths, idxs = load_streamlines_new(filename, idxs=range(n_streamlines), apply_affine=False, container='array', verbose=True)
    print(f"Total time: {time() - t0} sec.")

    print("")
    print("Using load_trk_numba.py")
    t0 = time()
    streamlines, header, lengths, idxs = load_streamlines_numba(filename, idxs=range(n_streamlines), apply_affine=False, container='array', verbose=True)
    print(f"Total time: {time() - t0} sec.")

    print("")
    filename_npy = filename[:-4] + '_no_resample.npy'
    print(f"Saving original streamlines in npz format to {filename_npy}")
    t0 = time()
    np.save(filename_npy, streamlines, allow_pickle=True)
    print(f"Total time: {time() - t0} sec.")

    print("")
    print(f"Loading {filename_npy}")
    t0 = time()
    S = np.load(filename_npy, allow_pickle=True)
    print(f"Total time: {time() - t0} sec.")

