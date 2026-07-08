Welcome to the OIC Toolkit Wiki!
======================================

The ``oic-toolkit`` is a consolidated Python package of useful image analysis functions 
developed by the Van Andel Institute Optical Imaging Core. The toolkit features a blend
of custom algorithms and workflows derived from scikit-image and scipy.

Features
--------

* Distance-based watershed algorithms
* Generating annotated plots to visualize segmentation results
* Stitching of tiled images
* ... and more!

Getting Started
---------------

Installation
~~~~~~~~~~~~

```bash
pip install oic-toolkit
```

Usage
~~~~~

```python
from oic_toolkit import *
```

Modules
^^^^^^^

The toolkit is organized into modular components that address specific bottlenecks in bioimage workflows:

* ``io.py``: Efficient reading and writing of standardized microscopy
  formats, including multi-resolution pyramidal OME-TIFFs.

* ``register.py``: Algorithms optimized for structural alignment, including sequential tissue section registration (e.g., "Swiss roll" intestinal tissue preparations).

* ``segment.py``: Repeatable tissue and cellular segmentation workflows leveraging morphology tools and watershed algorithms.

* ``display``: Standardized plotting functions and core array manipulations like downsampling, cropping, and contrast normalization.

Content Navigation
------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   auto_examples/index
   autoapi/index

