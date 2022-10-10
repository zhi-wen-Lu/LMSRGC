multi-GPU suffix array construction
=====

To compile, you need CMake 3.8 or greater and CUDA on the path.

 1. Create a build folder, preferably outside the source tree.
 2. cmake ../suffix-array/
 3. Optional: ccmake ../suffix-array/ (toggle settings, hit 'c', and 'g')
 4. make -j8

The compile option DGX1\_TOPOLOGY should be toggled (e.g. using ccmake) reflecting the setting of NUM\_GPUS in suffix\_array.cu.

With CUDA 10, there are many deprecation warnings caused by the warp intrinsics the multisplit by Ashkiani et al. uses (without _sync), 
however, there have been no errors so far. I have developed with CUDA 9 most of the time. Another multisplit implementation could be used instead.

Inputs up to a size of 2^32-2 can theoretically be sorted; on the DGX-1, about 3550 MB should work, 3520~MB have been successfully processed.

There is a bug with the merge-copy-detour-heuristics for worst-case inputs; these can be turned off by uncommenting the lines 305-334 of merge\_copy\_detour\_guide.hpp which may cost performance.

Some refactoring would be needed. 

"distrib-merge" means merging 2 distributed arrays that each are globally sorted.
"remerge" means merging multiple ranges of one distributed array, each consisting of several per-GPU locally sorted ranges, in parallel.

"gossip" contains a modified version of gossip
"multisplit" contains the multisplit by Ashkiani et al. with a wrapper class (dispatch_multisplit.cuh)

