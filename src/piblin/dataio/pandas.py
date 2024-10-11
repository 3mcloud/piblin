"""Conversion of cralds data structures into structured pandas DataFrames

DataFrames have become a foundational data structure throughout the
data science and machine learning communities. Well-supported
conversion between the analytical data analysis structures within
cralds and pandas data structures allows easy access to open-source
tools leveraging that assume dataframe structures.

Functions
-------
from_pandas: Converts pandas DataFrame into cralds MeasurementSet
"""
from copy import deepcopy
from typing import Union, Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import reduce
import importlib

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas must be installed to use convertor.")

from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.experiment import Experiment
from piblin.data.data_collections.measurement_set import MeasurementSet
from piblin.data.data_collections.experiment_set import ExperimentSet
from piblin.data.datasets.abc.dataset import Dataset


@dataclass
class DataSetHolder:
    """Convivence dataclass to hold information extracted from Datasets"""
    data: pd.DataFrame
    dep_name: str
    ind_name: str


def _process_dataset(
    dataset: Dataset, **kwargs: Dict[str, Union[str, float, int]]
) -> DataSetHolder:
    """Converts a single dataset into a DataSetHolder

    Parameters
    ----------
    dataset : Dataset
        cralds Dataset object
    **kwargs: Dict[str, Union[str, float, int]]
        Additionial measurement conditions to add to data.

    Returns
    -------
    extracted_dataset: DataSetHolder

    """
    tech = dataset.__class__.__module__ + "." + dataset.__class__.__name__
    dep_name = dataset.dependent_variable_names
    # TODO works for 1D but will break for 2D
    data_rows = []
    if isinstance(dataset, ZeroDimensionalDataset):
        ind_name = None
        dep_data = dataset.dependent_variable_data.item()
        data_rows = pd.DataFrame([{"cralds_cls": tech, **kwargs, dep_name[0]: dep_data}])
    else:
        independent_data = dict(zip(dataset.independent_variable_names, dataset.independent_variable_data))
        if isinstance(dataset.dependent_variable_data, list):
            dependent_data = dict(zip(dataset.dependent_variable_names, dataset.dependent_variable_data))
        else:
            dependent_data = {dataset.dependent_variable_names[0]: dataset.dependent_variable_data}
        ind_name = dataset.independent_variable_names
        data_rows = pd.DataFrame({**dependent_data, **independent_data})
        data_rows["cralds_cls"] = tech
        for k, v in kwargs.items():
            data_rows[k] = v
    return DataSetHolder(data_rows, dep_name, ind_name)


def _process_measurement(
    measurement: Measurement
) -> Tuple[List[DataSetHolder], List[str], Dict[str, Any]]:
    """Processes a single measurement into intermediate objects.

    Parameters
    ----------
    measurement : Measurement
        Specific measurement to be converted.

    Returns
    -------
    Tuple[List[DataSetHolder], List[str], Dict[str, Any]]
        Tuple with intermediate data structures for measurement data,
        measuremnt conditions with string and numeric values, and
        measurement conditions with values incompatible with pandas,
        respectively.
    """
    def attempt_pandas(test_item):
        try:
            df = pd.Series(test_item)
            return df.shape[0] == 1
        except:
            return False

    conditions_details = deepcopy(measurement.conditions)
    conditions_details.update(deepcopy(measurement.details))
    # Need a better test for non-Pandas compatible objects
    # This filters not enough stuff, 
    str_numeric_conds = {}
    other_conds = {}
    for k, v in conditions_details.items():
        if isinstance(v, (str, float, int)) or attempt_pandas(v):
            str_numeric_conds[k] = v
        else:
            other_conds[k] = v
    data_rows = []
    for dataset in measurement.datasets:
        data_rows.append(_process_dataset(dataset, **str_numeric_conds))
    # add class definition that tracks cralds class
    return data_rows, set(str_numeric_conds.keys()), other_conds


def _process_measurement_set(
    measurement_set: MeasurementSet
) -> Tuple[List[List[Dict[str, Any]]], List[List[str]], List[Dict[str, Any]]]:
    """Processes a MeasurmentSet into intermediate objects.

    Parameters
    ----------
    measurement : MeasurementSet
        Specific measurement to be converted.

    Returns
    -------
    Tuple[List[List[Dict[str, Any]]], List[List[str]], List[Dict[str, Any]]]
        Tuple with lists of intermediate data structures for measurement data,
        measuremnt conditions with string and numeric values, and
        measurement conditions with values incompatible with pandas,
        respectively.
    """
    data_rows = []
    all_conditions = []
    all_other = []
    for measurement in measurement_set.measurements:
        measure_rows, conditions, details = \
            _process_measurement(measurement)
        data_rows.append(measure_rows)
        all_conditions.append(conditions)
        all_other.append(details)
    return data_rows, all_conditions, all_other


def _to_pandas(
    cralds_object: Union[
        Dataset, Measurement, Experiment, MeasurementSet, ExperimentSet
    ],
    with_cralds_class: bool = True,
    concat: bool = False,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Converts a cralds data structure into a Pandas DataFrame.

    `_to_pandas` operates on all Dataset classes and organization classes
    within cralds.

    For MeasurementSet and ExperimentSet, a list of DataFrames are returned
    unless `concat=True`. Each DataFrame in the list corresponds to a single
    measurement with each set.

    Parameters
    ----------
    cralds_object :
        Union[Dataset, Measurement, Experiment, MeasurementSet, ExperimentSet]
        cralds data object to be converted
    with_cralds_class : bool, optional
        If True, a column corresponding to the source cralds Dataset class is
        added to the DataFrame. This simiplifies conversion back into a cralds
        data structure.
    concat : bool, optional
        If True, lists of DataFrames and concatenated into a single DataFrame.
        Used only with Experiment, MeasurementSet, and ExperimentSet conversion.

    Returns
    -------
    converted_data: Union[pd.DataFrame, List[pd.DataFrame]]
        DataFrame or list of DataFrames based on source data.

    Raises
    ------
    ImportError:
        Raised if `pandas` is not installed in environment.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas must be installed to use convertor.")

    def create_df(
        data: List[DataSetHolder],
        conditions: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        df = pd.concat([ds.data for ds in data], axis=0)
        ind_names = [ds.ind_name for ds in data if ds.ind_name is not None]
        if len(ind_names) > 1:
            ind_names = list(set(reduce(lambda a,b: a + b, ind_names)))
        elif len(ind_names) == 1:
            ind_names = ind_names[0]

        df.attrs = {
            "condition_columns": conditions,
            "details": details,
            "dependent_variable_names": list(set(reduce(lambda a,b: a + b, [ds.dep_name for ds in data]))),
            "independent_variable_names": ind_names
        }
        if not with_cralds_class:
            df = df.drop(columns="cralds_cls")
        return df

    # DataFrame doesn't respect Experiment vs Measurement
    if isinstance(cralds_object, ExperimentSet):
        cralds_object = cralds_object\
            .to_measurement_set(expand_replicates=True)
    elif isinstance(cralds_object, Experiment):
        cralds_object = cralds_object.to_measurement_set()

    if isinstance(cralds_object, Dataset):
        data_rows = _process_dataset(cralds_object)
        df = create_df([data_rows])
    elif isinstance(cralds_object, Measurement):
        data_rows, conditions, details = \
            _process_measurement(cralds_object)
        df = create_df(data_rows, conditions, details)
    elif isinstance(cralds_object, MeasurementSet):
        data_rows, conditions, details = _process_measurement_set(
            cralds_object
        )
        df = [
            create_df(data, cond, other_cond)
            for data, cond, other_cond in
            zip(data_rows, conditions, details)
        ]
        if concat:
            # merge attributes
            attrs = {}
            for k in [
                "condition_columns",
                "dependent_variable_names",
                "independent_variable_names",
            ]:
                attrs[k] = set().union(*[item.attrs[k] for item in df])
            # convert to
            attrs["details"] = reduce(
                lambda a, b: {**a, **b}, details
            )
            # Need to add measurement id to dfs to differentiate
            # duplicate measurements when in DataFrame
            def add_measurement_id(dataframe, n):
                dataframe["measurement_id"] = n
                return dataframe
            df = [add_measurement_id(df,n) for n, df in enumerate(df)]
            df = pd.concat(df, ignore_index=True)
            df.attrs = attrs
            df.attrs["condition_columns"].update({"measurement_id"})
    return df


def from_pandas(
    pandas_data: Union[
        pd.DataFrame,
        List[pd.DataFrame],
        Dict[str, pd.DataFrame]
    ],
    dependent_variable_names: Union[str, List[str]] = None,
    independent_variable_names: Union[str, List[str]] = None,
    condition_columns: Union[str, List[str], Dict[str, Any]] = None,
    details: Dict[str, Any] = None,
    cralds_cls: Optional[Union[str, type]] = None,
) -> MeasurementSet:
    """Converts Pandas DataFrame into cralds MeasurementSet

    Parameters
    ----------
    pandas_data : Union[
        pd.DataFrame,
        List[pd.DataFrame],
        Dict[str, pd.DataFrame]
    ]
        Pandas DataFrame to convert to cralds object.
    dependent_variable_name : Union[str, List[str]], optional
        Name of column corresponding to dependent variable. If None, `attrs` of
        DataFrame are parsed for dependent_variable_name key.
    independent_variable_names : Union[str, List[str]], optional
        Name of column(s) corresponding to independent variable(s).
        If None, `attrs` of DataFrame are parsed for
        independent_variable_name(s) key.
    condition_columns : Union[str, List[str], Dict[str, Any]], optional
        Name of column(s) corresponding to measurement conditions.
        If None, `attrs` of DataFrame are parsed for condition_columns.
    details : Dict[str, Any], optional
        Dictionary of conditions not stored in DataFrame. Values are
        added to all `Measurement` objects within `MeasurementSet`.
    cralds_cls : Union[str, type], optional
        cralds DataSet class to convert DataFrame into, either as a class object or a string representing the
        full package/module path. If None, `cralds_cls` column must exist in DataFrame.

    Returns
    -------
    converted_ms: MeasurementSet
        cralds MeasurementSet created from source data.

    Raises
    ------
    ValueError
        Raised if missing input argument definitions.
    """
    if cralds_cls is not None:
        if not isinstance(cralds_cls, str):
            # assume cralds_cls is a class object
            cralds_cls = cralds_cls.__module__ + "." + cralds_cls.__name__
        pandas_data["cralds_cls"] = cralds_cls
    attrs = pandas_data.attrs

    if condition_columns is not None:
        if isinstance(condition_columns, str):
            conditions = [condition_columns]
        else:
            conditions = condition_columns
    else:
        conditions = attrs.get("condition_columns", [])
        if conditions is None:
            conditions = []

    if details is None:
        details = attrs.get("details", {})

    if dependent_variable_names is None:
        dependent_variable_names = attrs.get("dependent_variable_names", None)
        # Bit goofy but need to prefer keyword argument
        if dependent_variable_names is None:
            raise ValueError(
                "No dependent_variable_name value was provided or defined"
                " in DataFrame."
            )

    if independent_variable_names is None:
        independent_variable_names = attrs.get("independent_variable_names")
        # Bit goofy but need to prefer keyword argument
        if independent_variable_names is None:
            raise ValueError(
                "No independent_variable_names value was provided or defined"
                " in DataFrame."
            )

    if isinstance(dependent_variable_names, str):
        def get_dep_name(columns):
            return dependent_variable_names
    else:
        def get_dep_name(columns):
            dep_name = [
                name for name in dependent_variable_names if name in columns
            ]
            return dep_name

    if isinstance(independent_variable_names, str):
        def get_ind_name(columns):
            return independent_variable_names
    else:
        def get_ind_name(columns):
            ind_name = [
                name for name in independent_variable_names if name in columns
            ]
            return ind_name

    def create_measurement(data, dataset_class, conditions):
        # remove NaN columns
        data = data.dropna(axis=1, how="all")
        dep_name = get_dep_name(data.columns)
        ind_name = get_ind_name(data.columns)
        dep_data = np.squeeze(data[dep_name].values)
        ind_data = data[ind_name].values
        # Convert 2d numpy array to list of 1D arrays
        ind_data = [row for row in ind_data.astype(float).T]
        # data can contain duplicate measurements so explicitly process
        # row-wise
        module_str, _, class_str = dataset_class.rpartition('.')
        dataset_class = getattr(importlib.import_module(module_str), class_str)
        if "ZeroDimensional" in dataset_class.__name__:
            cralds_data = dataset_class(
                dependent_variable_data=dep_data.astype(float),
                dependent_variable_names=dep_name,
            )
        else:
            cralds_data = dataset_class(
                dependent_variable_data=dep_data.astype(float),
                independent_variable_data=ind_data,
                dependent_variable_names=dep_name,
                independent_variable_names=ind_name,
            )
        if len(conditions) > 0:
            conditions = data.iloc[0][conditions].to_dict()
            conditions.update(details)
        else:
            conditions = None
        return Measurement([cralds_data], conditions=conditions, details=None)

    measurements = []
    for group, df in pandas_data.groupby(["cralds_cls", *conditions]):
        # if conditions are empty, group is a string
        if isinstance(group, str):
            group = [group]
        measurements.append(create_measurement(df, group[0], conditions))

    return MeasurementSet(measurements, merge_redundant=False)
