"""(C)orporate (R)esearch (A)nalytical (L)aboratory (D)ata (S)cience

Data science tools produced as part of the R&D Data Analytics (gALAxy)
project of the 3M Corporate Research Analytical Laboratory (CRAL).

This code forms the basis for a number of notebook and UI applications and
can be used in the IPython, Jupyter or Databricks environments for
interactive problem-solving. The package functionality which is intended to
be public is available by importing from the following subpackages of
cralds.

Packages
--------

piblin.data - Represent analytical datasets and hierarchical collections thereof.
piblin.dataio - Read/write hierarchically structured collections of datasets to/from files.
piblin.transform - Transform collections of datasets.
"""
import numpy as np

np.set_printoptions(threshold=0,
                    precision=4,
                    edgeitems=2,
                    suppress=True,
                    linewidth=10000,
                    sign="+",
                    floatmode="fixed")
