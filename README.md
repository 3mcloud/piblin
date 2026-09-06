# piblin

### A Framework for Measurement Data Science

[![PyPI Downloads](https://static.pepy.tech/badge/piblin)](https://pepy.tech/project/piblin)
[![Python 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](#)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

The `piblin` package is based on a number of abstractions related to
the collection and analysis of scientific data.  Its fundamental
concept is the `Dataset`, a mapping between independent and dependent
variables as typically obtained as the result of applying a scientific
measurement technique to a physical sample.  The dimensionality of
these datasets provides a simple approach to classifying them.

Beyond differences in dimensionality, specific types of dataset are
defined which allow for customization of the behaviour of `piblin`
for data collected by specific techniques.  For example, the
`Spectrum` and its many subtypes allow appropriate visualization of
datasets specific to given spectroscopic techniques.

Building on the dataset concept, scientific measurements are often
discussed in terms of a hierarchy of measurements, experiments and
projects, and this provides a natural language for collecting, editing
and analyzing scientific data.  The `piblin` package organizes
datasets via the concept of a `Measurement`, which joins a set of
conditions and details with corresponding datasets.  Sets of
measurements can be organized into `Experiment` objects, which
collect multiple measurements with the same conditions into sets of
repetitions, allowing for statistical analysis of variation among
them.  All of this organization is automatic, and can be controlled
by direct and simple editing of the metadata of measurements.
Hierarchically organized data can also be flattened for use in
machine-learning applications, and flat data can be converted into
hierarchical data in turn.

The final fundamental concept for performing data analytics on
experimental datasets is the `Transform`.  This is a procedure which
takes data as input and produces altered data as output.  For example
a spectroscopic dataset may be integrated over a region to yield the
area under a peak, a transformation from a 1D dataset to a scalar
value.  The set of transforms is infinite and `piblin` provides some
general and some specific classes that provide typical functionality.
As for file reading, the package makes it as easy as possible to add
a new transform, and a tutorial is included.

## Datasets

*Organizing independent and dependent data.*

A dataset collects points: pairs of dependent and independent variable
values,

$$y = f(\mathbf{x}).$$

The set of points for a dataset of dimensionality $n$ can be given a
more detailed definition,

$$\mathcal{P} = \{\,(\mathbf{x}, y) \in \{(x_0, x_1, \ldots, x_n) \in \mathbb{R}^n\} \times \{y \in \mathbb{R}\}\,\},$$

where the dimensionality $n$ is the number of independent variables
for each point.  Datasets are essentially context-free objects: they
carry the numerical relationship $y = f(\mathbf{x})$ without any
assumption about the physical technique that produced them.  Context
— sample identity, instrument conditions, processing history — is
attached at the `Measurement` level, one layer up.
