# Copyright (c) 2019 Pietro Astolfi, Emanuele Olivetti
# MIT License

import ipdb
import numpy as np
import nibabel as nib
import os
from time import time


def load_streamlines(trk_fn, idxs=None, apply_affine=True, container='list',
                     replace=False, verbose=False, load_twice=True):
    """Load streamlines from a .trk file. If a list of indices (idxs) is
    given, this function just loads and returns the requested
    streamlines, skipping all the non-requested ones.

    This function is sort of similar to nibabel.streamlines.load() but
    extremely FASTER. It is very convenient if you need to load only
    some streamlines in large tractograms. Like 100x faster than what
    you can get with nibabel.
    """

    if verbose:
        print("Loading %s" % trk_fn)

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    aff = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)

    if idxs is None:
        idxs = np.arange(nb_streamlines, dtype=np.int32)
    elif isinstance(idxs, int):
        if verbose:
            print('Sampling %s streamlines uniformly at random' % idxs)

        if idxs > nb_streamlines and (not replace):
            print('WARNING: Sampling with replacement')

        idxs = np.random.choice(np.arange(nb_streamlines), idxs,
                                replace=replace)
    else:  # useful in case of lists or range
        idxs = np.array(idxs)


    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    nb_bytes_float32 = np.dtype('<f').itemsize

    if verbose:
        print("Loading the whole data blob.")
        t0 = time()

    buffer = np.empty((os.path.getsize(trk_fn) // nb_bytes_float32), np.float32)
    with open(trk_fn, 'rb') as f:
        f.seek(1000)  # 1000 is the size of the header in bytes
        f.readinto(buffer)

    if verbose:
        print("%s values read in %s sec." % (buffer.size, time() - t0))

    if verbose:
        print("Parsing lengths of %s streamlines" % nb_streamlines)
        t0 = time()


    buffer_int32 = buffer.view(dtype=np.uint32)

    #####
    # Heuristic for extracting streamline lengths from a TRK file.
    ##
    # Leverage the fact that float32 numbers greater than ~1.4e-40
    # when viewed as uint32 are larger than 100000.
    # i.e., numbers < 100000 must be the 'length' values.
    lengths = buffer_int32[buffer_int32 < 100000]
    # Since float32(0) == int32(0), we may got false postives.
    # However, we can assume that length of a streamline is greater than 0.
    lengths = lengths[lengths > 0]
    if len(lengths) > nb_streamlines:
        # If we detected more 'length' values than there are streamlines,
        # we can fallback on the more robust (but slower) method.
        print("*Using fallback method*")
        lengths = np.empty(nb_streamlines, dtype=np.int32)
        pointer = 0
        for idx in range(len(lengths)):
            l = buffer_int32[pointer]
            lengths[idx] = l
            pointer += 1 + l * point_size + n_properties

    if verbose:
        print("%s sec." % (time() - t0))

    n_floats = lengths * point_size
    split_points = (n_floats + 1 + n_properties).cumsum() - n_floats - n_properties

    if verbose:
        print("Extracting %s streamlines" % len(idxs))
        if apply_affine:
            print("and applying the affine")

        t0 = time()

    # Load the streamline in an ArraySequence.
    seq = nib.streamlines.ArraySequence()
    seq._data = buffer
    seq._offsets = split_points
    seq._lengths = n_floats

    # Get the streamlines as a list instead of ArraySequence (to match the expected returned output).
    streamlines = [s.reshape(-1, 3) for s in seq[idxs]]

    if verbose:
        print("%s sec." % (time() - t0))
        t0 = time()

    if verbose:
        print("Converting all streamlines to the container %s" % container)
        t0 = time()

    if container == 'array':
        streamlines = np.array(streamlines, dtype=np.object)
    elif container == 'ArraySequence':
        streamlines = nib.streamlines.ArraySequence(streamlines)
    elif container == 'list':
        pass
    elif container == 'array_flat':
        streamlines = np.concatenate(streamlines, axis=0)
    else:
        raise Exception

    if verbose:
        print("%s sec." % (time() - t0))

    return streamlines, header, lengths[idxs], idxs


if __name__ == '__main__':

    np.random.seed(0)

    trk_fn = 'sub-100206_var-FNAL_tract.trk'
    trk_fn = 'sub-599469_var-10M_tract.trk'

    # idxs = np.random.choice(500000, 200000, replace=True)
    # idxs.sort()
    idxs = None  # This is for loading all streamlines
    # idxs = 1000

    streamlines, header, lengths, idxs = load_streamlines(trk_fn,
                                                              idxs,
                                                              apply_affine=True,
                                                              container='list',
                                                              verbose=True)

    print("Done.")
