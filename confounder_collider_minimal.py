"""Two factors, one control decision: confounder vs collider (minimal).

Plain top-to-bottom script (no functions). Two cases that invert: F1 as a
confounder of the F2 -> X relationship (must control) and F1 as a collider
(must not). The estimand is the causal loading of the return X on factor F2;
F1 is the candidate control. Each section draws one Monte Carlo sample at
seed 42 and reads two loadings (omit F1, then control F1) against the
population closed form.

Run: uv run python confounder_collider_minimal.py
"""

import numpy as np

SEED = 42
N = 1_000_000  # large enough that the loadings read off cleanly
b, a, c = 1.0, 0.8, 0.7  # true F2->X loading; F1<->F2 strength; F1<->X strength

# === Confounder: F1 -> F2, F1 -> X ===
# Omitting F1 leaves the backdoor F2 <- F1 -> X open; controlling F1 closes it.
rng = np.random.default_rng(SEED)
F1 = rng.standard_normal(N)
F2 = a * F1 + rng.standard_normal(N)
X = b * F2 + c * F1 + rng.standard_normal(N)

omit = np.linalg.lstsq(np.column_stack([F2, np.ones(N)]), X, rcond=None)[0][0]
ctrl = np.linalg.lstsq(np.column_stack([F2, F1, np.ones(N)]), X, rcond=None)[0][0]
omit_cf = b + a * c / (a**2 + 1)  # backdoor leaks in -> 1.3415
ctrl_cf = b  # X linear in (F2, F1) with independent noise -> exact

print("CONFOUNDER  (F1 -> F2, F1 -> X)")
print(f"  omit F1: {omit:.4f}  (closed form {omit_cf:.4f})  -> biased (backdoor open)")
print(f"  ctrl F1: {ctrl:.4f}  (closed form {ctrl_cf:.4f})  -> recovers true loading b = {b:.1f}")
print()

# === Collider: F2 -> F1, X -> F1 ===
# No backdoor, so F2 alone is unbiased; conditioning on F1 opens F2 -> F1 <- X.
rng = np.random.default_rng(SEED)
F2 = rng.standard_normal(N)
X = b * F2 + rng.standard_normal(N)
F1 = a * F2 + c * X + rng.standard_normal(N)

omit = np.linalg.lstsq(np.column_stack([F2, np.ones(N)]), X, rcond=None)[0][0]
ctrl = np.linalg.lstsq(np.column_stack([F2, F1, np.ones(N)]), X, rcond=None)[0][0]
omit_cf = b  # no backdoor to open
ctrl_cf = (b - a * c) / (c**2 + 1)  # exact closed form -> 44/149 = 0.2953

print("COLLIDER    (F2 -> F1, X -> F1)")
print(f"  omit F1: {omit:.4f}  (closed form {omit_cf:.4f})  -> unbiased (no backdoor)")
print(
    f"  ctrl F1: {ctrl:.4f}  (closed form {ctrl_cf:.4f})  -> biased toward zero (collider opened)"
)
