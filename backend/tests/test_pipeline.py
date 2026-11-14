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


from app.pipeline import fill_small_gaps
import numpy as np


def test_fill_small_gaps_fills_isolated_hole():
    img = np.ones((10, 10), dtype=np.float32) * 5.0
    img[4, 4] = np.nan  # a single isolated bad pixel, like the target's own peak
    filled, n_filled = fill_small_gaps(img, max_hole_pixels=6)
    assert n_filled == 1
    assert filled[4, 4] == pytest.approx(5.0)
    assert np.isfinite(filled).all()


def test_fill_small_gaps_leaves_large_holes_alone():
    img = np.ones((10, 10), dtype=np.float32) * 5.0
    img[2:8, 2:8] = np.nan  # a real 36-pixel masked region
    filled, n_filled = fill_small_gaps(img, max_hole_pixels=6)
    assert n_filled == 0
    assert np.isnan(filled[4, 4])


def test_fill_small_gaps_no_op_when_no_nans():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    filled, n_filled = fill_small_gaps(img)
    assert n_filled == 0
    np.testing.assert_array_equal(filled, img)


def test_fill_small_gaps_averages_multiple_neighbours():
    img = np.full((5, 5), np.nan, dtype=np.float32)
    img[1, 2] = 10.0
    img[2, 1] = 20.0
    img[2, 3] = 30.0
    img[3, 2] = 40.0
    # centre (2,2) is a 1-pixel hole surrounded by the four values above
    filled, n_filled = fill_small_gaps(img, max_hole_pixels=6)
    assert n_filled == 1
    assert filled[2, 2] == pytest.approx(25.0)  # mean of 10,20,30,40
