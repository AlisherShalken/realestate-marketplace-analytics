import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import abtest  # noqa: E402


def _ratio_sample(rng, n, p):
    den = rng.poisson(rng.gamma(1.5, 6, n)) + 1
    user_p = rng.beta(2, 2 / p - 2, n)
    num = rng.binomial(den, user_p)
    return num, den


def test_srm_balanced():
    assert abtest.srm_check(50_000, 50_100) > 0.05


def test_srm_broken():
    assert abtest.srm_check(50_000, 51_500) < 0.001


def test_delta_method_matches_bootstrap():
    rng = np.random.default_rng(1)
    nc, dc = _ratio_sample(rng, 5000, 0.07)
    nt, dt = _ratio_sample(rng, 5000, 0.07)
    var_delta = abtest.ratio_var(nc, dc)
    w = rng.poisson(1, size=(3000, len(nc)))
    boot = (w @ nc) / (w @ dc)
    assert var_delta == pytest.approx(boot.var(), rel=0.1)


def test_delta_method_fpr_on_aa():
    # на А/А тестах доля ложных прокрасов должна быть около alpha
    rng = np.random.default_rng(7)
    hits = 0
    n_sim = 400
    for _ in range(n_sim):
        nc, dc = _ratio_sample(rng, 2000, 0.07)
        nt, dt = _ratio_sample(rng, 2000, 0.07)
        hits += abtest.delta_ratio_test(nc, dc, nt, dt).p_value < 0.05
    assert 0.02 < hits / n_sim < 0.08


def test_cuped_reduces_variance_and_keeps_mean():
    rng = np.random.default_rng(3)
    x = rng.gamma(2, 3, 10_000)
    y = 0.8 * x + rng.normal(0, 2, 10_000)
    y_adj, theta = abtest.cuped(y, x)
    assert theta == pytest.approx(0.8, abs=0.05)
    assert y_adj.mean() == pytest.approx(y.mean())
    assert y_adj.var() < 0.5 * y.var()


def test_holm():
    rejected, adj = abtest.holm([0.01, 0.04, 0.03, 0.2])
    assert list(rejected) == [True, False, False, False]
    assert adj[0] == pytest.approx(0.04)
