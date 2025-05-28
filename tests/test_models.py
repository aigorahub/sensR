from senspy import BetaBinomial


def test_beta_binomial_mean():
    model = BetaBinomial(alpha=2.0, beta=2.0, n=10)
    assert model.mean() == 0.5


def test_beta_binomial_pmf_sum():
    model = BetaBinomial(alpha=1.0, beta=1.0, n=5)
    s = sum(model.pmf(k) for k in range(6))
    assert abs(s - 1.0) < 1e-12
