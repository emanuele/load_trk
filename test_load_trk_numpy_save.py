import numpy as np
from load_trk import load_streamlines
from time import time

if __name__=='__main__':

    n_streamlines = int(4e6)
    
    filename = 'data/sub-599469_var-10M_tract.trk'
    streamlines, header, lengths, idxs = load_streamlines(filename, idxs=range(n_streamlines), apply_affine=False, container='array', verbose=True)

    filename_npy = filename[:-4] + '_no_resample.npy'
    print(f"Saving original streamlines in npz format to {filename_npy}")
    t0 = time()
    np.save(filename_npy, streamlines, allow_pickle=True)
    print(f"{time() - t0} sec.")

    print(f"Loading {filename_npy}")
    t0 = time()
    S = np.load(filename_npy, allow_pickle=True)
    print(f"{time() - t0} sec.")

    
