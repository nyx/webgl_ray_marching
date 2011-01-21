Synopsis
========
Efficiently render animated distorted implicit 'blobby' surfaces in modern web browsers.  Render with a ray-marching algorithm implemented in WebGL environment (essentially OpenGL ES 2.0 Javascript bindings) that executes for the most part on the GPU.

Notes
=====
Two triangles are rendered with a shader that implements a ray-marching algorithm that finds zero-crossings of an implicit surface defined by the summation of three point emitters with various radii.

A ray for each pixel marches down a ray at fixed steps until an interval is found that contains a zero-crossing in summed field sample, then a fixed number of binary search steps refine the intersection point estimate.

The point emitter positions are animated in Javascript along with the value of time.

Space is distorted by a 4D smooth noise function running on the GPU.  Textures that serve as lookup tables for the GPU noise functions are generated procedurally in Javascript. 