import numpy as np

from dynaflux.data.preprocessing import TrafficScaler, impute_missing


def test_imputation_and_scaler_roundtrip():
    x = np.array([[[1.0]], [[0.0]], [[3.0]], [[np.nan]]], dtype=np.float32)
    filled, mask = impute_missing(x, zero_as_missing=True)
    assert mask.sum() == 2
    assert np.isfinite(filled).all()
    scaler = TrafficScaler("robust").fit(filled)
    restored = scaler.inverse_transform(scaler.transform(filled))
    assert np.allclose(restored, filled, atol=1e-5)
