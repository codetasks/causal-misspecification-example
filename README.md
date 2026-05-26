# Causal Misspecification Example

A self-contained pedagogical notebook on factor model misspecification: what happens
to a factor loading when you omit versus control for a second factor `F1`, across the
five roles that second factor can play.

## View the widgets (live, in the browser)

Both HTML files are self-contained (inline CSS and JavaScript, no dependencies), so they
render live through htmlpreview without any setup:

- [Bar-chart widget](https://htmlpreview.github.io/?https://github.com/codetasks/causal-misspecification-example/blob/main/explainer_causal_factors_two_factor.html)
  — true effect vs OLS estimate as two bars.
- [Scatter widget](https://htmlpreview.github.io/?https://github.com/codetasks/causal-misspecification-example/blob/main/explainer_causal_factors_two_factor_scatter.html)
  — sample cloud with the true-effect and OLS slope lines (added-variable plot when controlling).

GitHub renders raw `.html` as source, so the links above use htmlpreview to run the page.
You can also download either file and open it directly in any browser.

## Contents

- `explainer_causal_factors_two_factor.ipynb` — one control decision for a second
  factor `F1` across all five roles (confounder, collider, mediator, neutral predictor,
  driver of factors only). Each case prints the omit and control loadings plus the
  direction of bias, and closes with a qualitative decision table.
- `confounder_collider_minimal.py` — minimal runnable script trimmed to the two roles
  that invert: `F1` as a confounder (must control) and as a collider (must not). Each
  case reports the omit and control loadings against the population closed form.
- `explainer_causal_factors_two_factor.html` — interactive widget (bar chart): flip `F1`
  between confounder and collider, tick the control on or off, and watch the verdict invert.
- `explainer_causal_factors_two_factor_scatter.html` — the same widget rendered as a scatter
  of a sample (n = 1200, seed 42) with two slope lines: the dashed green true-effect slope `β`
  and the solid OLS-fit slope. Ticking the control switches to an added-variable
  (partial-regression) plot, so the controlled coefficient is shown honestly as the slope
  through the residual cloud.

## Run

```bash
pip install numpy
python confounder_collider_minimal.py          # minimal script
# or, for the full walkthrough:
pip install jupyter
jupyter notebook explainer_causal_factors_two_factor.ipynb
```

The notebook is committed without cell outputs and runs top to bottom from a fresh
kernel. Stochastic cells are seeded (42) for reproducibility. The collider closed form
is `(beta - a*c)/(c^2 + 1)`; at `beta=1, a=0.8, c=0.7` it equals `44/149 = 0.295`.
