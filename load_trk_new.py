# Copyright (c) 2019 Pietro Astolfi, Emanuele Olivetti
# MIT License

import numpy as np
import nibabel as nib
from struct import unpack
from time import time
# import sys


def get_length_numpy(f, nb_bytes_int32=4):
    """Parse an int32 from a file. NumPy version.

    NOTE: this implementation is 20x slower in Python >3.2 than
    previous versions because of:
    https://github.com/numpy/numpy/issues/13319

    ...but if you use np.frombuffer() instead it is fast again!
    https://github.com/SciTools/iris/pull/3791
    """
    # return np.fromfile(f, np.int32, 1)[0]
    return np.frombuffer(f.read(nb_bytes_int32), np.int32, 1)[0]


def get_length_struct(f, nb_bytes_int32=4, int32_fmt='<i'):
    """Parse an int32 from a file. struct.unpack() version.
    """
    return unpack(int32_fmt, f.read(nb_bytes_int32))[0]


def get_length_from_bytes(f, nb_bytes_int32=4, byteorder='little'):
    """Parse an int32 from a file. int.from_bytes() version.

    NOTE: int.from_bytes() is available only from Python >3.2
    """
    return int.from_bytes(f.read(nb_bytes_int32), byteorder=byteorder)


def load_streamlines(trk_fn, idxs=None, apply_affine=True,
                     container='list', replace=False, verbose=False):
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
        idxs = np.arange(nb_streamlines, dtype=int)
    elif isinstance(idxs, int):
        if verbose:
            print('Sampling %s streamlines uniformly at random' % idxs)

        if idxs > nb_streamlines and (not replace):
            print('WARNING: Sampling with replacement')

        idxs = np.random.choice(np.arange(nb_streamlines), idxs,
                                replace=replace)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    nb_bytes_float32 = np.dtype('<f').itemsize

    if verbose:
        print("Parsing lenghts of %s streamlines" % nb_streamlines)
        t0 = time()

    lengths = np.empty(nb_streamlines, dtype=int)

    # In order to reduce the 20x increase in time when reading small
    # amounts of bytes with NumPy and Python >3.2, we use two
    # different implementations of the function that parses 4 bytes
    # into an int32:
    # if float(sys.version[:3]) > 3.2:
    #     get_length = get_length_from_bytes
    # else:
    #     get_length = get_length_numpy

    # The following function is OKish for all versions of Python:
    get_length = get_length_struct

    with open(trk_fn, 'rb') as f:
        f.seek(header_size)
        for idx in range(nb_streamlines):
            l = get_length(f)
            lengths[idx] = l
            jump = point_bytes * l + properties_bytes
            f.seek(jump, 1)

    if verbose:
        print("%s sec." % (time() - t0))

    # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size
    # n_floats = lengths * point_size + n_properties
    n_floats = lengths * point_size  # better because it skips properties, if they exist

    if verbose:
        print("Extracting %s streamlines with the desired id" % len(idxs))
        t0 = time()

    streamlines = []
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            # s = np.fromfile(f, np.float32, n_floats[idx])  # very slow in in Python >3.2
            # s = np.frombuffer(f.read(nb_bytes_float32 * n_floats[idx]), np.float32, n_floats[idx])  # faster in Python >3.2 than np.fromfile
            # The following is even faster:
            # s = np.empty(n_floats[idx], np.float32)
            # f.readinto(s)
            # s.resize(lengths[idx], point_size)
            # And this is even slightly better:
            s = np.empty((lengths[idx], point_size), np.float32)
            f.readinto(s)
            # remove scalars if present:
            if n_scalars > 0:
                s = s[:, :3]

            if apply_affine:
                s = nib.affines.apply_affine(aff, s)

            streamlines.append(s)

        if verbose:
            print("%s sec." % (time() - t0))

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
