"""
Module: roi

Classes
-------
LinearRegion
CompoundRegion
"""
from typing import List, Union


class LinearRegion(object):
    """Linear region of a 1D space.

    Parameters
    ----------
    region : list
        The values at the edges of the region.
    name : str
        An identifier for this linear region.

    Attributes
    ----------
    region -> list
        The ordered values at the edges of the region.
    name -> str
        An identifier for this linear region.

    Methods
    -------
    contains - Determine whether this region contains the given point.
    """
    def __init__(self, region: List[Union[int, float]], name: str = None):
        self._region = region
        self.__validate_region()
        self._name = name

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        self._region = region
        self.__validate_region()

    def __validate_region(self):
        """Validate the values of this linear region."""
        if len(self.region) != 2:
            raise ValueError("Region has 2 values only.")
        for i, value in enumerate(self.region):
            self.region[i] = float(value)
        self.region.sort()

    @property
    def region_min(self):
        return self._region[0]

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

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def contains(self, point):
        """Determine whether this region contains the given point.

        Parameters
        ----------
        point : float or int
            The value for which to determine containment.

        Returns
        -------
        bool
            True iff this region contains the specified point.
        """
        if self._region[0] <= point <= self._region[1]:
            return True

        return False

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


class CompoundRegion(object):
    """A collection of linear regions comprising a compound region.

    The order of linear regions is irrelevant, but as they are not
    hashable they are stored as a list.

    Parameters
    ----------
    linear_regions : list of LinearRegion
        The regions of this compound region.
        By default the compound region has no regions.

    Attributes
    ----------
    linear_regions : list of LinearRegion
        The regions of this compound region.

    Methods
    -------
    append - Append a specified region to this compound region.
    extend - Append multiple specified regions to this compound region.
    remove - Remove a specified region from this compound region.
    contains - Determine whether this compound region contains a specified point.
    """
    def __init__(self, linear_regions: list = None):
        if linear_regions is None:
            self._linear_regions = []
        else:
            self._linear_regions = linear_regions

    @staticmethod
    def from_data(regions: list,
                  names: list = None,
                  name_base: str = "region"):
        """Create a set of linear regions from numerical values.

        Parameters
        ----------
        regions : list of list
        names : list of str
        name_base : str

        Returns
        -------
        CompoundRegion
        """

        if names is None:
            names = []
            for i, region in enumerate(regions):
                names.append(name_base + " " + str(i+1))

        linear_regions = []
        for region, name in zip(regions, names):
            linear_regions.append(LinearRegion(region, name))

        return CompoundRegion(linear_regions)

    @property
    def linear_regions(self):
        return self._linear_regions

    def append(self, linear_region):
        self.linear_regions.append(linear_region)

    def extend(self, linear_regions):
        self.linear_regions.extend(linear_regions)

    def remove(self, linear_region):
        self.linear_regions.remove(linear_region)

    def contains(self, point):
        """Determine whether this compound region contains a specified point.

        Parameters
        ----------
        point : float or int
            The value for which to determine containment.

        Returns
        -------
        bool
            True iff this compound region contains the specified point.
        """
        if len(self.linear_regions) == 0:
            return False

        elif len(self.linear_regions) == 1:
            return self.linear_regions[0].contains(point)

        else:

            for linear_region in self.linear_regions:
                if linear_region.contains(point):
                    return True

            return False

    def __len__(self):
        return len(self._linear_regions)

    def __getitem__(self, position):
        return self.linear_regions[position]

    def __setitem__(self, key, value):
        self.linear_regions[key] = value

    def __str__(self):
        """Create a human-readable representation of this object."""
        str_rep = f"{self.__class__.__name__}\n" + "-" * len(self.__class__.__name__) + "\n"
        for linear_region in self.linear_regions:
            str_rep += f"{linear_region}\n"

        return str_rep[:-1] + "\n"

    def __repr__(self):
        """Create an eval()-able representation of this object."""
        str_rep = self.__class__.__name__ + "("
        str_rep += "linear_regions=["
        for linear_region in self.linear_regions:
            str_rep += repr(linear_region) + ", "
        return str_rep[:-2] + "])"

    def __eq__(self, other):
        """Determine if this compound region is equal to another."""
        if self is other:
            return True

        for linear_region in self.linear_regions:
            if linear_region not in other.compound_region:
                return False

        for linear_region in other.compound_region:
            if linear_region not in self.linear_regions:
                return False

        return True

    def __cmp__(self, other):
        """Ensure comparisons other than equality are not implemented."""
        return NotImplemented
