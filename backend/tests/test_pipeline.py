"""Offline unit tests for pipeline.py functions that don't need the
network (grouping/capping logic). See test_pipeline_live.py for the
network-dependent full-pipeline tests."""
import pytest

from app.pipeline import cap_group_indices


def test_cap_no_op_when_under_limit():
    assert cap_group_indices([3, 7, 9], max_count=5) == [3, 7, 9]


def test_cap_none_means_unlimited():
    assert cap_group_indices(list(range(20)), max_count=None) == list(range(20))


def test_cap_zero_means_unlimited():
    assert cap_group_indices(list(range(20)), max_count=0) == list(range(20))


def test_cap_keeps_first_and_last():
    idxs = list(range(100, 120))  # 20 items
    capped = cap_group_indices(idxs, max_count=5)
    assert capped[0] == idxs[0]
    assert capped[-1] == idxs[-1]
    assert len(capped) <= 5


def test_cap_single_item_picks_middle():
    idxs = [10, 11, 12, 13, 14]
    assert cap_group_indices(idxs, max_count=1) == [12]


def test_cap_preserves_sorted_order():
    idxs = list(range(0, 50, 2))  # 25 items, already sorted
    capped = cap_group_indices(idxs, max_count=7)
    assert capped == sorted(capped)
    assert len(capped) <= 7


@pytest.mark.parametrize("n, cap", [(30, 5), (30, 15), (7, 15), (1, 5)])
def test_cap_never_exceeds_requested_and_never_exceeds_input(n, cap):
    idxs = list(range(n))
    capped = cap_group_indices(idxs, max_count=cap)
    assert len(capped) <= min(n, cap)
    assert len(capped) >= 1
    assert set(capped) <= set(idxs)
