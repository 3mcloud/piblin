# """
#
# Protected visualization methods are tested via the subclasses of Dataset.
# The dataset class has the following properties:
#
# num_independent_variables
# num_dependent_variables
# dependent_variable_data
# dependent_variable_name
# independent_variable_data
# independent_variable_names
#
# Methods
# flatten
# flatten_dependent_variables
# unflatten
# unflatten_dependent_variables
# decode_column_labels
# visualize
# produce_color_map
# style_axes
#
# init
# eq
# cmp
# str
# repr
# """
#
# import numpy as np
# from numpy import array  # not unused - do not delete
# import pytest
# from cralds.data.datasets.dataset import Dataset, LabelError, DimensionalityError, VisualizationError
# from cralds.data import Scalar
#
#
# @pytest.fixture()
# def scalar_dataset():
#     return Dataset(dependent_variable_data=0.0)
#
#
# @pytest.fixture()
# def actual_scalar():
#     return Scalar(0.0)
#
#
# @pytest.fixture()
# def scalar_dataset_with_units():
#     return Dataset(dependent_variable_data=0.0,
#                    dependent_variable_unit="units")
#
#
# def test_flatten_not_implemented(scalar_dataset):
#     with pytest.raises(NotImplementedError):
#         scalar_dataset.flatten()
#
#
# def test_unflatten_not_implemented(scalar_dataset):
#     with pytest.raises(NotImplementedError):
#         scalar_dataset.unflatten(None, None)
#
#
# @pytest.fixture()
# def one_d_dataset_single_point():
#     return Dataset(dependent_variable_data=0.0,
#                    independent_variable_data=0.0)
#
#
# @pytest.fixture()
# def one_d_dataset_single_point_with_units():
#     return Dataset(dependent_variable_data=0.0,
#                    dependent_variable_unit="units",
#                    independent_variable_data=0.0,
#                    independent_variable_units=["units"])
#
#
# @pytest.fixture()
# def one_d_dataset_multiple_points():
#     return Dataset(dependent_variable_data=[0.0, 1.0],
#                    independent_variable_data=[0.0, 0.0])
#
#
# def test_ambiguity_is_resolved():
#     """Ensure a bare scalar and length-1 list result in the same object.
#
#     When the user provides a scalar python value as the dependent variable
#     data of a dataset, it should result in the creation of a scalar. The same
#     is true if a length-1 list is provided as the dependent variable data
#     without accompanying independent_variable_data.
#     For one-dimensional datasets, the same object must be produced whether
#     the dependent or independent variables are provided as scalar or length-1
#     list variables, in all combinations.
#     """
#     assert Dataset(dependent_variable_data=0.0) == Dataset(dependent_variable_data=[0.0])
#
#     assert Dataset(dependent_variable_data=0.0,
#                    independent_variable_data=0.0) == \
#            Dataset(dependent_variable_data=[0.0],
#                    independent_variable_data=[0.0])
#
#     assert Dataset(dependent_variable_data=0.0,
#                    independent_variable_data=0.0) == \
#            Dataset(dependent_variable_data=0.0,
#                    independent_variable_data=[0.0])
#
#     assert Dataset(dependent_variable_data=0.0,
#                    independent_variable_data=0.0) == \
#            Dataset(dependent_variable_data=[0.0],
#                    independent_variable_data=0.0)
#
#
# def test_initializer_errors():
#
#     # the dimensionality of the dependent array must be the number of independent variables
#     with pytest.raises(DimensionalityError):
#         Dataset(dependent_variable_data=[0.0],
#                 independent_variable_data=[[0.0], [0.0]])
#
#     with pytest.raises(DimensionalityError):
#         Dataset(dependent_variable_data=[0.0, 0.0])
#
#     with pytest.raises(LabelError):
#         Dataset(dependent_variable_data=0.0,
#                 independent_variable_data=0.0,
#                 independent_variable_names=["x0", "x1"])
#
#
# def test_independent_variable_names():
#     """Assert the behavior of the validation method for names."""
#     test_str = "i"
#
#     # a scalar (0D dataset) has no independent variable names
#     assert Dataset(0.0).independent_variable_names == []
#     assert Dataset([0.0]).independent_variable_names == []
#
#     # a single-point 1D dataset has one independent variable name
#     assert Dataset(0.0, 0.0).independent_variable_names == ["i_1"]
#     assert Dataset([0.0], 0.0).independent_variable_names == ["i_1"]
#     assert Dataset(0.0, [0.0]).independent_variable_names == ["i_1"]
#     assert Dataset([0.0], [0.0]).independent_variable_names == ["i_1"]
#     # a multi-point 1D dataset has one independent variable name
#     assert Dataset([0.0, 1.0], [0.0, 0.0]).independent_variable_names == ["i_1"]
#
#     # specific name for single-point 1D dataset
#     assert Dataset(0.0, 0.0, [test_str]).independent_variable_names == [test_str]
#     assert Dataset([0.0], 0.0, [test_str]).independent_variable_names == [test_str]
#     assert Dataset(0.0, [0.0], [test_str]).independent_variable_names == [test_str]
#     assert Dataset([0.0], [0.0], [test_str]).independent_variable_names == [test_str]
#     # specific name for multi-point 1D dataset
#     assert Dataset([0.0, 1.0], [0.0, 0.0], ["i"]).independent_variable_names == ["i"]
#
#
# def test_basic_2d():
#     """Create a simple 4-pixel image, e.g.
#
#     This is a future-looking test, a 2D dataset class will be added.
#     """
#     two_d_dataset = Dataset([[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]])
#     assert two_d_dataset.num_dependent_variables == 1
#     assert two_d_dataset.num_independent_variables == 2
#     assert two_d_dataset.dependent_variable_name == "variable"
#     assert two_d_dataset.independent_variable_names == ["i_1", "i_2"]
#
#
# def test_basic_3d():
#     """Create a simple volumetric image, e.g.
#
#     This is a future-looking test, a 3D dataset will be added.
#     """
#     three_d_dataset = Dataset([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
#                               [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
#
#     assert three_d_dataset.num_dependent_variables == 1
#     assert three_d_dataset.num_independent_variables == 3
#     assert three_d_dataset.dependent_variable_name == "variable"
#     assert three_d_dataset.independent_variable_names == ["i_1", "i_2", "i_3"]
#
#
# def test_scalar_units(scalar_dataset,
#                       scalar_dataset_with_units,
#                       actual_scalar):
#
#     assert scalar_dataset.dependent_variable_unit is None
#     assert actual_scalar.unit == scalar_dataset.dependent_variable_unit
#     assert actual_scalar.unit == actual_scalar.dependent_variable_unit
#     assert scalar_dataset_with_units.dependent_variable_unit == "units"
#
#
# def test_oned_units(one_d_dataset_single_point,
#                     one_d_dataset_single_point_with_units):
#     assert one_d_dataset_single_point.dependent_variable_unit is None
#     assert one_d_dataset_single_point_with_units.dependent_variable_unit == "units"
#     assert one_d_dataset_single_point_with_units.independent_variable_units == ["units"]
#
#
# def test_scalar_dependent_variable_name_setter(scalar_dataset):
#     scalar_dataset.dependent_variable_name = "test"
#     assert scalar_dataset.dependent_variable_name == "test"
#
#
# def test_scalar_dependent_variable_name_arg(scalar_dataset):
#     assert Dataset(dependent_variable_data=0.0,
#                    dependent_variable_name="x").dependent_variable_name == "x"
#
#
# def test_scalar_dependent_variable_data_access():
#     assert Dataset(dependent_variable_data=0.0).dependent_variable_data == [0.0]
#     assert Dataset(dependent_variable_data=0.0).dependent_variable_data == 0.0
#     assert Dataset(dependent_variable_data=[0.0]).dependent_variable_data == [0.0]
#     assert Dataset(dependent_variable_data=[0.0]).dependent_variable_data == 0.0
#
#
# def test_scalar_dependent_variable_setter(scalar_dataset):
#     scalar_dataset.dependent_variable_data = 1.0
#     assert scalar_dataset == Dataset([1.0])
#
#
# def test_scalar_independent_variable_data_access(scalar_dataset):
#     assert np.array_equal(scalar_dataset.independent_variable_data, np.array([]))
#     assert np.array_equal(scalar_dataset.independent_variable_data, [])
#
#
# def test_scalar_property_values(scalar_dataset):
#     assert scalar_dataset.num_independent_variables == 0
#     assert scalar_dataset.num_dependent_variables == 1
#     assert scalar_dataset.independent_variable_names == []
#     assert(eval(repr(scalar_dataset))) == scalar_dataset
#
#
# def test_dataset_oned_independent_names_setter(one_d_dataset_single_point):
#     one_d_dataset_single_point.independent_variable_names = ["test"]
#     assert one_d_dataset_single_point.independent_variable_names == ["test"]
#
#     one_d_dataset_single_point.independent_variable_names = "test"
#     assert one_d_dataset_single_point.independent_variable_names == ["test"]
#
#
# def test_dataset_oned_independent_names_set_to_none(one_d_dataset_single_point):
#     """Ensure that setting the independent names to None resets them to default."""
#     altered_dataset = one_d_dataset_single_point
#     altered_dataset.independent_variable_names = "test"
#     altered_dataset.independent_variable_names = None
#
#     assert altered_dataset == Dataset(0.0, 0.0)
#
#
# def test_dataset_oned_inependent_names_error(one_d_dataset_single_point):
#     with pytest.raises(LabelError):
#         one_d_dataset_single_point.independent_variable_names = ["", ""]
#
#
# def test_dataset_oned_independent_variable_data_setter(one_d_dataset_single_point):
#     dataset = Dataset(0.0, 1.0)
#     dataset.independent_variable_data = [0.0]
#     assert dataset == one_d_dataset_single_point
#
#
# def test_dataset_oned_single_point(one_d_dataset_single_point):
#     assert one_d_dataset_single_point.num_independent_variables == 1
#     assert one_d_dataset_single_point.num_dependent_variables == 1
#     assert len(one_d_dataset_single_point.independent_variable_names) == 1
#     assert eval(repr(one_d_dataset_single_point)) == one_d_dataset_single_point
#
#
# def test_one_d_dataset_multiple_points_property_values(one_d_dataset_multiple_points):
#     assert one_d_dataset_multiple_points.num_dependent_variables == 1
#     assert one_d_dataset_multiple_points.num_independent_variables == 1
#     assert len(one_d_dataset_multiple_points.independent_variable_names) == 1
#     #assert eval(repr(one_d_dataset_multiple_points)) == one_d_dataset_multiple_points
#
#
# def test_one_line_str(scalar_dataset,
#                       one_d_dataset_single_point,
#                       one_d_dataset_multiple_points):
#     """Ensure the one line str is one line."""
#     assert scalar_dataset.one_line_str().count("\n") == 0
#     assert one_d_dataset_single_point.one_line_str().count("\n") == 0
#     assert one_d_dataset_multiple_points.one_line_str().count("\n") == 0
#
#
# def test_visualize_not_implemented(scalar_dataset):
#     with pytest.raises(NotImplementedError):
#         scalar_dataset.visualize()
#
#
# def test_comparisons_not_implemented(scalar_dataset):
#     with pytest.raises(TypeError):
#         assert scalar_dataset > scalar_dataset
#
#     with pytest.raises(TypeError):
#         assert scalar_dataset < scalar_dataset
#
#     with pytest.raises(TypeError):
#         assert scalar_dataset >= scalar_dataset
#
#     with pytest.raises(TypeError):
#         assert scalar_dataset <= scalar_dataset
#
#
# def test_equals():
#     """Check the implementation of equality, including labelling."""
#     assert Dataset(0.0) == Dataset(0.0)
#     assert Dataset(0.0) != Dataset(1.0)
#     assert Dataset(0.0, dependent_variable_name="y") != Dataset(0.0, dependent_variable_name="f(x)")
#
#     assert Dataset([0.0], [0.0]) == Dataset([0.0], [0.0])
#     assert Dataset([0.0], [0.0]) != Dataset([1.0], [0.0])
#     assert Dataset([0.0], [0.0]) != Dataset([0.0], [1.0])
#
#     assert Dataset([0.0], [0.0], independent_variable_names=["i"]) != Dataset([0.0], [0.0])
#
#
# def test_dataset_visualize(actual_scalar):
#     actual_scalar.visualize()
#
#
# def test_dataset_visualize_error(actual_scalar):
#     with pytest.raises(VisualizationError):
#         actual_scalar.visualize(axes=[])
