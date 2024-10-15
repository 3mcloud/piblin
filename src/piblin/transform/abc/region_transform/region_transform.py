from typing import List
from abc import ABC
import piblin.transform.abc.transform as transform
import piblin.data.datasets.roi as roi


class RegionTransform(transform.Transform, ABC):
    """Transform restricted to specific regions.

    Certain old_transforms are either applied to a restricted set
    of data points based on their independent variables, or have
    parameters computed based on a restricted set of variables
    which are then used to transform an entire dataset.
    In all cases, the transform has an associated set of regions.
    As this pattern is common, this base class exists to prevent
    significant repetition of code.

    Parameters
    ----------
    compound_region : CompoundRegion
        The compound region over which to apply the transform.
        By default an empty compound region is created.

    Attributes
    ----------
    compound_region -> roi.CompoundRegion
        The linear regions of this region transform.
    data_independent_parameters -> List of float or List of int
        The data-independent parameters (region bounds) of this transform.
    data_dependent_parameters -> List of object
        The data-dependent parameters (an empty list) of this transform.
    is_frozen -> bool
        Whether this transform's data-dependent parameters are to be recomputed or not.

    Methods
    -------
    from_linear_region
        Create a region transform from a single linear region.
    from_data
        Create a region transform from lists of ranges and names.
    """
    DEFAULT_RANGE_COLOR = "orange"
    DEFAULT_ALPHA = 0.5

    def __init__(self,
                 compound_region: roi.CompoundRegion = None,
                 *args,
                 **kwargs):

        if compound_region is None:
            compound_region = roi.CompoundRegion()

        super().__init__(
            data_independent_parameters=[compound_region],
            *args,
            **kwargs
        )

        self._compound_region = compound_region

    @property
    def compound_region(self) -> roi.CompoundRegion:
        """The linear regions of this region transform."""
        return self._compound_region

    @property
    def data_independent_parameters(self) -> List[float]:
        """Return the (flattened) data-independent linear region parameters of this region transform."""
        data_independent_parameters = []
        for linear_region in self.compound_region.linear_regions:
            data_independent_parameters.extend([linear_region.region_min,
                                                linear_region.region_max])
        return data_independent_parameters

    @data_independent_parameters.setter
    def data_independent_parameters(self, data_independent_parameters: List[float]) -> None:
        """Set the data-independent linear region parameters of this region transform.

        Parameters
        ----------
        data_independent_parameters : List of float
            The new (flattened) data-independent linear region parameters of this region transform.
        """

        if len(data_independent_parameters) % 2 != 0:
            raise ValueError(f"Number of parameters to region transform must be even: "
                             f"received {len(data_independent_parameters)}")

        for region_index in range(0, len(data_independent_parameters), 2):
            self.compound_region.linear_regions[region_index].min_value = data_independent_parameters[region_index]
            self.compound_region.linear_regions[region_index].min_value = data_independent_parameters[region_index + 1]

    @classmethod
    def from_linear_region(cls, linear_region: roi.LinearRegion) -> "RegionTransform":
        """Create a region transform from a single linear region.

        Parameters
        ----------
        linear_region : roi.LinearRegion
            The linear region from which to construct the region transform.

        Returns
        -------
        RegionTransform
            A region transform constructed from the supplied linear region.
        """
        return cls(compound_region=roi.CompoundRegion([linear_region]))

    @classmethod
    def from_data(cls,
                  regions: list = None,
                  names: list = None,
                  name_base: str = "region") -> "RegionTransform":
        """Create a region transform from lists of ranges and names.

        Parameters
        ----------
        regions : list of list of float
            The regions of the region transform's compound region.
        names : list of str
            The names of the region transform's regions.
        name_base : str
            A base name from which to construct names for the region transform's regions.

        Returns
        -------
        RegionTransform
            A region transform constructed from the supplied regions and names.
        """
        compound_region = roi.CompoundRegion.from_data(regions=regions,
                                                       names=names,
                                                       name_base=name_base)
        return cls(compound_region)

    def visualize_transform_parameters(self, axes):
        """Visualize the parameters of this region transform.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to visualize this region transform's parameters.
        """
        for linear_region in self.compound_region:
            axes.axvspan(linear_region.region_min,
                         linear_region.region_max,
                         alpha=self.DEFAULT_ALPHA,
                         facecolor=self.DEFAULT_RANGE_COLOR)

    def __repr__(self):
        """Create an eval-able representation of this transform."""
        str_rep = self.__class__.__name__ + "("
        str_rep += "linear_regions="
        for region in self.compound_region:
            str_rep += repr(region) + ", "
        return str_rep[:-2] + ")"

    def __str__(self):
        str_rep = f"{self.__class__.__name__}\n"
        str_rep += "Regions:"
        for region in self.compound_region:
            str_rep += f"{str(region)}\n"
        return str_rep
