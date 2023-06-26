``SubchunkableApplyFlow``
========================

.. autofunction:: zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow.build_subchunkable_apply_flow

Divisibility Requirement
------------------------

At each subchunking level, the ``processing_`` parameters of ``SubchunkableApplyFlow`` are subject to divisiblity requirements. Specifically, the processing chunk at each level **must** evenly divide the ``crop`` and ``blend`` padded processing chunk of the level above. Furthermore, the top level processing chunk **must** evenly divide the ``bbox``.

This requirement exists because ``SubchunkableApplyFlow`` recursively subchunks the padded ROI into the processing chunks of the level below. ``auto_divisiblity`` exists to automatically find the correct processing chunk sizes, but it may have an adverse impact on performance.

.. note::

   If calculating the ``processing_chunk_sizes`` by hand, make sure to multiply the ``crop_pad`` and ``blend_pad`` by 2 before adding it to the ``processing_chunk_size``.

Note on Architecture
--------------------

This subsection is intended as an in-depth discussion of the **SubchunkableApplyFlow** architecture.

All of the volumes that ``zetta_utils`` expects to work with are backed by a chunked storage. This means that if we want to use padding of any form in our processing, whether that is blending or cropping, we run into two major issues:

#. Assuming that the processing chunks (hereafter referred to as **punks**) are exactly the size of the backend chunks (**bunks**), adding even a single pixel of padding in XY incurs a 9x read penalty since the neighbouring chunks must be read.
#. Since the writing is also done to chunks, we have a race condition in writes when using blending. 9, 27, or even more (if a bunk is small) punks with padding can overlap in a bunk, which results in a race condition.

Solving problem #1 is the motivation for why ``SubchunkableApplyFlow`` needs to exist, and solving #2 is the motivation for its architecture.

To solve the first problem, we can use caching. However, naive chunking means that cache locality cannot be guaranteed. A single node in a large run might see more than 10,000 tasks, and a cache might only be large enough for about 100 bunks, meaning that a downloaded chunk will be evicted after 10 tasks since 9 chunks need to be downloaded for each task.

Subchunking so that each node sees a "superchunk" that should entirely fit in memory guarantees that we are using caching, thereby *massively* reducing the amount of read required. In fact, we can even use disk-backed caching on a "super-superchunk", with each "superchunk" fitting in memory if we want to reduce the overhead from caching even further.

In addition, the subchunking allows task generation to be naturally parallelised since each node is responsible for splitting up the chunks it received into subchunks. This is a bigger boon than it might seem at the beginning: if generating a single task takes a millisecond, then scheduling 1e8 tasks takes 28 hours.

Solving the second problem is a little harder: the easy solution is to just emit the tasks in 9 or 27 waves and add to the existing output (with blending weights) at every write, but this has its own issues:

#. Emitting tasks in waves is very inefficient since there will always be straggler tasks. A thousand nodes should not be held back because one node was preempted.
#. If a node gets preempted, each bunk needs to be tracked to see if the existing data should be ignored.
#. Each write now incurs a read call as well.
#. Each node requires 9 or 27 read and write calls.

Thus, the better solution is actually to just write to different temporary layers and sum with the weights at the end. This might seem expensive, but thanks to subchunking we can use local storage or memory as temporary layers, and just write to remote temporary layers for the top level. (This relies on the fact that local I/O is practically free, even when chunked.)

From here on, we will refer to the bottom level processing chunk and the bottom level backend chunk as the *L0* punk and *L0* bunk, respectively. The superchunks that consist of *L(n)* punks will be denoted *L(n+1)* punks, with the entire ROI to be processed being one level higher than the highest level.

Whenever blending is used in ``SubchunkableApplyFlow``, 8 temporary layers are created for checkerboarding in X, Y, Z. Each of these temporary layers have a smaller bunk than the original layer, at half of the punk given, and aligned to the punks. This ensures that there are no race conditions within one of these 8 layers, as long as ``blend_crop`` is less than half of the punk. The tasks can be emitted all at once, and once the processing tasks are done, there is a reduction task for each original bunk that collects all the processing tasks that intersect that bunk and sums them all with the correct weights. When blending is not used, a single temporary layer is created to solve the problem of the punk not being aligned to the bunk, or the bunk being too large (resulting in a race condition).

So how does this checkerboarding interact with the subchunking? ``SubchunkableApplyFlow`` is fully recursive, and here is the rundown for any given level:

#. Start with the level *N+1* bunk and crop_padded punk. We are given the level *N* punk, crop_pad and blend_pad.
#. Divide each crop_padded *N+1* punk into level *N* punks (divisibility is checked).
#. Make 8 temporary layers, each with bunks that are aligned to the crop_padded *N+1* punk and with half the size of level N punks.
#. Pad the level *N* punks, and recursively schedule and execute level *N* tasks.
#. Reduce the results for each level *N+1* bunk.
#. Return

In practice, to save time, the reduction happens not for each bunk, but for each superbunk consisting of bunks that have been aggregated up to some size, specified by ``max_reduction_chunk_size``. Note that the *L0* ``crop_pad`` is handled by the function or the operation.

This design makes it possible to use arbitrary cropping and blending at each level.
