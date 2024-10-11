import pytest
from piblin.data.datasets.roi import LinearRegion


@pytest.fixture
def named_region():
    return LinearRegion([0, 1], "basic")


@pytest.fixture
def anonymous_region():
    return LinearRegion([0, 1])


def test_name_access_and_setter(named_region):
    named_region.name = "name"
    assert named_region.name == "name"


@pytest.fixture
def inverse_order_named_region():
    return LinearRegion([1, 0], "inverse")


def test_repr(named_region, anonymous_region):
    assert eval(repr(named_region)) == named_region
    assert eval(repr(anonymous_region)) == anonymous_region


def test_equals(named_region, anonymous_region):
    """Ensure only numerical content defines equality."""
    assert named_region == anonymous_region


def test_sorting(named_region, inverse_order_named_region):
    assert named_region == inverse_order_named_region


def test_sorting_min(named_region, inverse_order_named_region):
    named_region.region_min = 2
    inverse_order_named_region.region_min = 2
    assert named_region == inverse_order_named_region


def test_sorting_max(named_region, inverse_order_named_region):
    named_region.region_max = -1
    inverse_order_named_region.region_max = -1
    assert named_region == inverse_order_named_region
