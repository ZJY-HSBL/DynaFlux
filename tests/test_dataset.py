import numpy as np

from dynaflux.data.dataset import MultiPeriodTrafficDataset, PeriodSpec


def test_period_alignment():
    data = np.arange(2300 * 3, dtype=np.float32).reshape(2300, 3, 1)
    ds = MultiPeriodTrafficDataset(data, PeriodSpec())
    item = ds[0]
    assert item["hour"].shape == (12, 3, 1)
    assert item["day"].shape == (12, 3, 1)
    assert item["week"].shape == (12, 3, 1)
    assert item["target"].shape == (12, 3, 1)
