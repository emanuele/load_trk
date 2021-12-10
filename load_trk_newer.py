# Copyright (c) 2019 Pietro Astolfi, Emanuele Olivetti
# MIT License

import os
import numpy as np
import nibabel as nib
from struct import unpack
from time import time


def get_length_struct(f, nb_bytes_int32=4, int32_fmt='<i'):
    """Parse an int32 from a file. struct.unpack() version.
    """
    return unpack(int32_fmt, f.read(nb_bytes_int32))[0]


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

    nb_bytes_uint32 = np.dtype('<u4').itemsize
    nb_bytes_float32 = np.dtype('<f').itemsize

    if verbose:
        print("Parsing lenghts of %s streamlines" % nb_streamlines)
        t0 = time()

    offsets = np.empty(nb_streamlines, dtype=int)
    lengths = np.empty(nb_streamlines, dtype=np.int32)


    #####
    # Heuristic for extracting streamline lengths from a TRK file.
    ##
    # Leverage the fact that float32 numbers greater than ~1.4e-40
    # when viewed as uint32 are larger than 100000.
    # i.e., numbers < 100000 must be the 'length' values.
    # Since float32(0) == int32(0), we may got false postives.
    # However, we can assume that length of a streamline is greater than 0.
    chunk_size = 256 * 1024 ** 2  # 256 Mb
    buffer = np.empty((chunk_size // nb_bytes_uint32), np.uint32)
    with open(trk_fn, 'rb') as f:
        f.seek(1000)  # 1000 is the size of the header in bytes

        cnt = 0
        chunk_id = 0
        while True:
            nbytes_read = f.readinto(buffer)
            if not nbytes_read:
                break

            nb_int32_read = nbytes_read // nb_bytes_uint32
            offsets_ = np.where(np.logical_and(buffer[:nb_int32_read] < 100000, buffer[:nb_int32_read] > 0))[0]
            offsets[cnt:cnt+len(offsets_)] = offsets_
            offsets[cnt:cnt+len(offsets_)] *= nb_bytes_uint32
            offsets[cnt:cnt+len(offsets_)] += chunk_id * chunk_size
            lengths[cnt:cnt+len(offsets_)] = buffer[offsets_]

            cnt += len(offsets_)
            chunk_id += 1

    lengths = np.array(lengths)
    offsets += header_size + nb_bytes_uint32
    index_bytes = offsets

    if verbose:
        print("%s sec." % (time() - t0))
        t0 = time()

    if len(offsets) > nb_streamlines:
        # If we detected more 'length' values than there are streamlines,
        # we can fallback on the more robust (but slower) method.
        print("*Using fallback method*")

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

        lengths = np.empty(nb_streamlines, dtype=np.int32)
        with open(trk_fn, 'rb') as f:
            f.seek(header_size)
            for idx in range(nb_streamlines):
                l = get_length(f)
                lengths[idx] = l
                jump = point_bytes * l + properties_bytes
                f.seek(jump, 1)

        # position in bytes where to find a given streamline in the TRK file:
        index_bytes = lengths * point_bytes + properties_bytes + length_bytes
        index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size

        if verbose:
            print("%s sec." % (time() - t0))

    if verbose:
        print("Extracting %s streamlines with the desired id" % len(idxs))
        t0 = time()

    seq = nib.streamlines.ArraySequence()

    seq._data = np.empty((lengths.sum(), point_size), np.float32)
    seq._lengths = lengths
    seq._offsets = np.cumsum(np.pad(lengths[:-1], (1, 0), "constant"))  # Prepend 0

    with open(trk_fn, 'rb') as f:
        for idx, offset in zip(idxs, seq._offsets):
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            f.readinto(seq._data[offset:offset+lengths[idx]])

    if verbose:
        print("%s sec." % (time() - t0))
        print("Casting ArraySequence to list of ndarray")
        t0 = time()

    # Get the streamlines as a list instead of ArraySequence (to match the expected returned output).
    streamlines = [s[:, :3] for s in seq]  # NB: This is slow.

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
