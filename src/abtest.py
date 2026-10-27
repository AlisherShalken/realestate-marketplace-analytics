"""Небольшой набор функций для анализа A/B тестов, которыми пользуюсь в ноутбуках."""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    control: float
    test: float
    diff: float
    rel_diff: float
    ci_low: float
    ci_high: float
    p_value: float

    def __repr__(self):
        return (f"control={self.control:.5f} test={self.test:.5f} "
                f"diff={self.rel_diff:+.2%} CI(abs)=[{self.ci_low:.5f}; {self.ci_high:.5f}] "
                f"p={self.p_value:.4f}")


def srm_check(n_control, n_test, expected_share=0.5):
    """Sample ratio mismatch — хи-квадрат на размеры групп. p < 0.001 — повод не смотреть тест дальше."""
    total = n_control + n_test
    expected = [total * (1 - expected_share), total * expected_share]
    return stats.chisquare([n_control, n_test], expected).pvalue


def _z_result(m_c, m_t, var_diff, alpha):
    diff = m_t - m_c
    se = np.sqrt(var_diff)
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    q = stats.norm.ppf(1 - alpha / 2)
    return TestResult(m_c, m_t, diff, diff / m_c, diff - q * se, diff + q * se, p)


def ttest(x_control, x_test, alpha=0.05):
    """Welch t-test для поюзерной метрики."""
    x_control = np.asarray(x_control, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    var_diff = x_control.var(ddof=1) / len(x_control) + x_test.var(ddof=1) / len(x_test)
    return _z_result(x_control.mean(), x_test.mean(), var_diff, alpha)


def ratio_var(num, den):
    """Дисперсия оценки sum(num)/sum(den) через дельта-метод (юзер — единица рандомизации)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cov = np.cov(num, den, ddof=1)[0, 1]
    return (vx / my**2 - 2 * mx * cov / my**3 + mx**2 * vy / my**4) / n


def delta_ratio_test(num_c, den_c, num_t, den_t, alpha=0.05):
    """Тест для ratio-метрики (например, контакты / просмотры)."""
    r_c = np.sum(num_c) / np.sum(den_c)
    r_t = np.sum(num_t) / np.sum(den_t)
    return _z_result(r_c, r_t, ratio_var(num_c, den_c) + ratio_var(num_t, den_t), alpha)


def linearize(num, den, ratio_control):
    """Линеаризация ratio-метрики: L = num - R_c * den. После неё можно применять обычный t-test и CUPED."""
    return np.asarray(num, dtype=float) - ratio_control * np.asarray(den, dtype=float)


def cuped(y, x_pre):
    """
    CUPED: y' = y - theta * (x - mean(x)).
    theta считаем по объединенной выборке обеих групп, иначе можно занести смещение.
    """
    y = np.asarray(y, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / x_pre.var(ddof=1)
    return y - theta * (x_pre - x_pre.mean()), theta


def bootstrap_ratio(num_c, den_c, num_t, den_t, n_boot=2000, alpha=0.05, seed=0):
    """Поюзерный бутстрап разницы ratio-метрик. Медленнее дельта-метода, но удобен как проверка."""
    rng = np.random.default_rng(seed)
    num_c, den_c = np.asarray(num_c), np.asarray(den_c)
    num_t, den_t = np.asarray(num_t), np.asarray(den_t)
    nc, nt = len(num_c), len(num_t)
    # пуассоновский бутстрап — веса вместо ресемплинга индексов, так быстрее
    wc = rng.poisson(1, size=(n_boot, nc))
    wt = rng.poisson(1, size=(n_boot, nt))
    rc = (wc @ num_c) / (wc @ den_c)
    rt = (wt @ num_t) / (wt @ den_t)
    diffs = rt - rc
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs, (lo, hi), max(p, 1 / n_boot)


def holm(p_values, alpha=0.05):
    """Поправка Холма — возвращает маску отвергнутых гипотез и скорректированные p-value."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj < alpha, adj


def mde(std, n_per_group, alpha=0.05, power=0.8):
    """Минимальный детектируемый эффект (в абсолютных единицах) для двух равных групп."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * std * np.sqrt(2 / n_per_group)


def sample_size(std, effect, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * std / effect) ** 2))
