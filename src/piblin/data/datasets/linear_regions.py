from collections import namedtuple, UserList
from typing import List, Tuple, Union


class LinearRegion(object):
    """A region bounded by minimum and maximum values.

    Parameters
    ----------
    region
    """
    Edge = namedtuple(typename="Edge",
                      field_names=["min", "max"],
                      defaults=[0, 0])

    def __init__(self,
                 region: List[Tuple[Union[int, float],
                                    Union[int, float]]],
                 name: str = None):

        self._num_dimensions = len(region)
        self._region = tuple([self.Edge(values[0], values[1])
                              for values in region])

    @property
    def num_dimensions(self):
        return self._num_dimensions

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        self._region = region

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def contains(self, point: List) -> bool:
        """Determine whether this region contains the given point."""

        if len(point) != self.num_dimensions:
            raise ValueError()

        else:
            for dimension in range(self.num_dimensions):
                if not self.region[dimension].min < point[dimension] < self.region[dimension].max:
                    return False

            return True

    def __eq__(self, other):
        """Determine whether this linear region is equal to another.

        This equality operator only considers numerical equality; the name
        of the linear regions is ignored.
        """
        if self.region != other.region:
            return False
        else:
            return True

    def __str__(self):
        """Create a human-readable representation of this linear region."""
        str_rep = self.__class__.__name__
        if self.name is not None:
            str_rep += f" ({self.name}) "
        str_rep += (f"between {self.region_min:.3f} "
                    f"and {self.region_max:.3f}")
        return str_rep

    def __repr__(self):
        """Create an eval()-able representation of this linear region."""
        str_rep = self.__class__.__name__ + "("
        str_rep += str(self.region)
        if self.name is not None:
            str_rep += ", name=\"" + self.name + "\""
        return str_rep + ")"


class OneDLinearRegion(LinearRegion):

    def __init__(self,
                 region: Tuple[Union[int, float],
                               Union[int, float]],
                 name: str = None):

        super().__init__(region=[region],
                         name=name)

    @property
    def region(self) -> Tuple[Union[int, float],
                              Union[int, float]]:
        """"""
        return self._region[0]

    @region.setter
    def region(self, region: "Edge") -> None:
        self._region = region

    @property
    def region_min(self):
        return self.region.min

    @property
    def min(self):
        return self.region_min

    @region_min.setter
    def region_min(self, value):
        self.region[0] = value
        self.__validate_region()

    @property
    def region_max(self):
        return self._region[1]

    @property
    def max(self):
        return self.region_max

    @region_max.setter
    def region_max(self, value):
        self.region[1] = value
        self.__validate_region()


class TwoDLinearRegion(LinearRegion):
    ...


class ThreeDLinearRegion(LinearRegion):
    ...


class CompoundRegion(UserList):
    """A collection of linear regions comprising a compound region."""
    def __init__(self, linear_regions: List[LinearRegion] = None):

        super().__init__(linear_regions)

        if not all(linear_region.num_dimensions ==
                   linear_regions[0].num_dimensions
                   for linear_region in linear_regions[1:]):

            raise ValueError(
                "Number of dimensions must match for all linear regions."
            )

        self._linear_regions = linear_regions

    def from_data(self):
        ...

    def contains(self, point) -> bool:
        """Determine whether this compound region contains a point.

        Parameters
        ----------

        Returns
        -------
        bool
            True iff this compound region contains the specified point.
        """
        for linear_region in self:
            if linear_region.contains(point):
                return True

        return False
