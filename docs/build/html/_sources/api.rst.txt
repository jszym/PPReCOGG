API Documentation
=================

gaborExtract
------------
The gaborExtract module is responsible for efficiently extracting
Gabor means and stdevs from a random sampling of pixels, using multiple
window sizes and Gabor orientations. These operations are GPU-accelerated
on platforms with compatible GPUs. Features are stored in HDF5 files.

.. automodule:: pprecogg.gaborExtract
   :members:


classifyFeatures
------------
Given features extracted from the gaborExtract model, this module 
parses feature files and classifies sets of unknown pixel features 
as belonging to known pixel feature-sets using a k-nearest-neighbors
approach.

.. automodule:: pprecogg.classifyFeatures
   :members: