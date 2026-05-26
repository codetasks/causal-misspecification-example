"""Two factors, one control decision: confounder vs collider (minimal).

Trimmed from explainer_causal_factors_two_factor.ipynb to the two cases that
invert: F1 as a confounder (must control) and F1 as a collider (must not).
The estimand is the causal loading of the return X on factor F2; F1 is the
candidate control. Each case draws one Monte Carlo sample at seed 42 and reads
off two loadings (omit F1, then control F1) against the population closed form.

Run: uv run python confounder_collider_minimal.py
"""

import numpy as np

SEED = 42
N = 1_000_000  # large enough that the loadings read off cleanly


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Seeded NumPy Generator; seed is explicit and logged, per the reproducibility rule."""
    return np.random.default_rng(seed)


def ols_slope(y, *regressors):
    """OLS slope on the FIRST regressor; remaining regressors are controls. Intercept added."""
    R = np.column_stack(list(regressors) + [np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(R, y, rcond=None)
    return coef[0]


def confounder(b: float = 1.0, a: float = 0.8, c: float = 0.7) -> None:
    """F1 -> F2, F1 -> X: omitting F1 leaves the backdoor F2 <- F1 -> X open.

    Closed form omit:    b + a*c/(a^2 + 1)   (= 1.3415 at the defaults)
    Closed form control: b                   (X linear in (F2, F1), independent noise)
    """
    rng = set_seed()
    F1 = rng.standard_normal(N)
    F2 = a * F1 + rng.standard_normal(N)
    X = b * F2 + c * F1 + rng.standard_normal(N)

    biased = ols_slope(X, F2)  # backdoor left open
    recovered = ols_slope(X, F2, F1)  # backdoor closed by controlling F1

    omit_cf = b + a * c / (a**2 + 1)
    ctrl_cf = b
    print("CONFOUNDER  (F1 -> F2, F1 -> X)")
    print(f"  omit F1: {biased:.4f}  (closed form {omit_cf:.4f})  -> biased (backdoor open)")
    print(
        f"  ctrl F1: {recovered:.4f}  (closed form {ctrl_cf:.4f})  -> recovers true loading b = {b:.1f}"
    )


def collider(b: float = 1.0, a: float = 0.8, c: float = 0.7) -> None:
    """F2 -> F1, X -> F1: conditioning on the collider opens the spurious path F2 -> F1 <- X.

    Closed form omit:    b                   (F2 exogenous, no backdoor)
    Closed form control: (b - a*c)/(c^2 + 1) (= 44/149 = 0.2953 at the defaults)

    The simplified form equals b only when c = 0 (the X -> F1 arrow vanishes, so F1 is no
    longer a collider) or in the knife edge a = -b*c; a collider with c != 0 biases the
    loading for essentially any a.
    """
    rng = set_seed()
    F2 = rng.standard_normal(N)
    X = b * F2 + rng.standard_normal(N)
    F1 = a * F2 + c * X + rng.standard_normal(N)

    unbiased = ols_slope(X, F2)  # no backdoor: F2 alone is already correct
    biased = ols_slope(X, F2, F1)  # conditioning on the collider opens the path

    omit_cf = b
    ctrl_cf = (b - a * c) / (c**2 + 1)
    print("COLLIDER    (F2 -> F1, X -> F1)")
    print(f"  omit F1: {unbiased:.4f}  (closed form {omit_cf:.4f})  -> unbiased (no backdoor)")
    print(
        f"  ctrl F1: {biased:.4f}  (closed form {ctrl_cf:.4f})  -> biased toward zero (collider opened)"
    )


if __name__ == "__main__":
    confounder()
    print()
    collider()
