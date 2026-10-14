import numpy as np

from pvil.eval.harmony import run_harmony


def _toy(n=300, d=20, seed=0):
    rng = np.random.default_rng(seed)
    batch = np.repeat(["a", "b", "c"], n // 3)
    x = rng.normal(size=(n, d)) + (batch == "b")[:, None] * 2.0
    return x, batch


def test_shape_and_info():
    x, batch = _toy()
    z, info = run_harmony(x, batch, theta=2, lamb=1, sigma=0.1, max_iter_harmony=100, device="cpu")
    assert z.shape == x.shape
    assert info["harmonypy_version"] == "0.2.0"
    assert info["params"] == {"theta": 2, "lamb": 1, "sigma": 0.1, "max_iter_harmony": 100, "random_state": 0}
    assert 1 <= info["n_iter"] <= 100
    assert info["n_batches"] == 3


def test_reduces_batch_shift():
    x, batch = _toy()
    z, _ = run_harmony(x, batch, theta=2, lamb=1, sigma=0.1, max_iter_harmony=100, device="cpu")
    gap = lambda m: np.linalg.norm(m[batch == "b"].mean(0) - m[batch == "a"].mean(0))
    assert gap(z) < 0.5 * gap(x)


def test_deterministic_on_cpu():
    x, batch = _toy()
    z1, _ = run_harmony(x, batch, theta=2, lamb=1, sigma=0.1, max_iter_harmony=100, device="cpu")
    z2, _ = run_harmony(x, batch, theta=2, lamb=1, sigma=0.1, max_iter_harmony=100, device="cpu")
    np.testing.assert_allclose(z1, z2)
