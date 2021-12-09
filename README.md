# load_trk
A set of fast loaders for TRK (Trackvis) files. They also allow random access to the streamlines.

The test file, sub-100206_var-FNAL_tract.trk (500K streamlines, 680Mb), is available from here: https://nilab.cimec.unitn.it/people/olivetti/data/sub-100206_var-FNAL_tract.trk

A much larger test tractogram, sub-599469_var-10M_tract.trk (10M streamlines, 2.9Gb), is available from here: https://nilab.cimec.unitn.it/people/olivetti/data/sub-599469_var-10M_tract.trk

This is a simple test for timing the loading of 4 million streamlines out of a TRK file with 10 million streamlines, across the three available implementations. As lower bound, the same data is loaded with numpy.load() after converting it to a numpy.array:

```
$ python test_load_trk_numpy_save.py
Using load_trk.py
Loading data/sub-599469_var-10M_tract.trk
Parsing lenghts of 10000000 streamlines
8.771811723709106 sec.
Extracting 4000000 streamlines with the desired id
33.70155715942383 sec.
Converting all streamlines to the container array
0.5225656032562256 sec.
Total time: 43.513148069381714 sec.

Using load_trk_new.py
Loading data/sub-599469_var-10M_tract.trk
Parsing lenghts of 10000000 streamlines
9.227784633636475 sec.
Extracting 4000000 streamlines with the desired id
11.10650897026062 sec.
Converting all streamlines to the container array
0.5689153671264648 sec.
Total time: 22.209201335906982 sec.

Using load_trk_numba.py
Loading data/sub-599469_var-10M_tract.trk
Loading the whole data blob.
751859555 values read in 0.8997900485992432 sec.
Parsing lengths of 10000000 streamlines
1.0455410480499268 sec.
Extracting 10000000 streamlines
3.3145549297332764 sec.
Converting all streamlines to the container array
0.5106325149536133 sec.
Total time: 7.029937982559204 sec.

Saving original streamlines in npz format to data/sub-599469_var-10M_tract_no_resample.npy
Total time: 21.208060264587402 sec.

Loading data/sub-599469_var-10M_tract_no_resample.npy
Total time: 6.182585954666138 sec.
```
