# Causal Misspecification Example

A self-contained pedagogical notebook on factor model misspecification: what happens
to a factor loading when you omit versus control for a second factor `F1`, across the
five roles that second factor can play.

## Contents

- `explainer_causal_factors_two_factor.ipynb` — one control decision for a second
  factor `F1` (confounder, collider, mediator, neutral predictor, driver of factors
  only). Each case prints the omit and control loadings plus the direction of bias,
  and closes with a qualitative decision table.

## Run

```bash
pip install numpy pandas
jupyter notebook explainer_causal_factors_two_factor.ipynb
```

The notebook is committed without cell outputs and runs top to bottom from a fresh
kernel. Stochastic cells are seeded for reproducibility.
