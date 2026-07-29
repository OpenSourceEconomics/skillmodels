# Linear predict optimization

## When the linear predict is used

At model setup time, `is_all_linear` checks whether every latent factor's transition
function name belongs to `{"linear", "constant"}`. This is an all-or-nothing decision:
if even one factor uses a nonlinear transition (e.g. `translog`), the entire model falls
back to the unscented predict.

The check happens in `get_maximization_inputs`, where the predict function is selected
via `functools.partial`. When the linear path is chosen, extra keyword arguments
(`latent_factors`, `constant_factor_indices`, `n_all_factors`) are bound at setup time
so the predict function has the same call signature as the unscented variant.

## Why it is faster and uses less memory

The unscented predict generates $2n + 1$ sigma points (where $n$ is the number of latent
factors), transforms each one through the transition function, then recovers predicted
means and covariances from weighted statistics. Its QR decomposition operates on a
matrix of shape $(3n + 1) \times n$: the $2n + 1$ weighted deviation rows plus $n$ rows
for the shock standard deviations.

The linear predict skips sigma-point generation entirely. Because the transition is
linear, the predicted mean is just a matrix--vector product, and the predicted
covariance follows from the standard linear Gaussian formula. Its QR decomposition
operates on a $(2n) \times n$ matrix: $n$ rows from the propagated Cholesky factor and
$n$ rows for the shocks. The reduction from $3n + 1$ to $2n$ rows speeds up the QR step
and removes all sigma-point overhead.

The memory savings can be more important than the speed gains. The unscented path
materialises $2n + 1$ sigma points for every observation and mixture component, and
JAX's automatic differentiation retains intermediate buffers for the backward pass. The
linear path replaces all of this with a single matrix multiply whose memory footprint
scales with $n^2$ rather than with the number of sigma points times the number of
observations. On memory-constrained GPUs this can be the difference between fitting the
model and running out of memory.

## Building F and c

The linear predict assembles a transition matrix $F$ of shape
$(n_\text{latent}, n_\text{all})$ and a constant vector $c$ of length $n_\text{latent}$
from the `trans_coeffs` dictionary. Here $n_\text{all}$ includes both latent and
observed factors.

For each latent factor $i$:

- **Linear factor**: `trans_coeffs[factor]` is a 1-d array whose last element is the
  intercept and whose preceding elements are the coefficients on all factors (latent and
  observed). Row $i$ of $F$ is set to `coeffs[:-1]` and $c_i$ is set to `coeffs[-1]`.
- **Constant factor**: row $i$ of $F$ is the unit vector $e_i$ (identity row) and
  $c_i = 0$, so the factor value is simply carried forward.

The implementation uses a stack-then-mask approach: all coefficient arrays are stacked
into a single matrix (with zero-padded rows for constant factors), an identity matrix
provides the constant-factor rows, and `jnp.where` selects between them using a boolean
mask. This avoids per-element `.at[i].set()` calls and conditional branching, producing
a cleaner trace for JAX's compiler.

Three construction strategies were benchmarked (loop with conditional `.at[i].set()`,
stack-then-mask with `jnp.where`, and index-scatter with pre-separated sub-matrices).
All three produced identical XLA graphs and showed no meaningful runtime difference
(~6.3--6.7 ms per call on CPU, 4-factor model, 5000 observations), confirming that the
construction is fully resolved at trace time. The stack-then-mask variant was kept for
its cleaner, more idiomatic JAX style.

## Mean prediction

The mean prediction incorporates anchoring, which rescales factors to a common metric
across periods. Let $s^{\text{in}}$ and $c^{\text{in}}$ be the input-period scaling
factors and constants, and $s^{\text{out}}$ and $c^{\text{out}}$ the output-period
counterparts. The steps are:

1. **Anchor** the input states: $x^a = x \odot s^{\text{in}} + c^{\text{in}}$.
1. **Concatenate** observed factors to form the full state vector
   $\tilde{x} = [x^a, x^{\text{obs}}]$.
1. **Apply the linear transition**: $y^a = \tilde{x}\, F^\top + c$.
1. **Un-anchor** to get the predicted states:
   $\hat{x} = (y^a - c^{\text{out}}) \oslash s^{\text{out}}$.

## Covariance prediction (square-root form)

skillmodels maintains covariances in square-root (upper Cholesky) form throughout. Let
$R$ denote the current upper Cholesky factor so that $P = R^\top R$. The linear predict
propagates $R$ as follows.

Define the effective transition matrix

$$
G = \operatorname{diag}(1 / s^{\text{out}})\; F_{\text{latent}}\;
    \operatorname{diag}(s^{\text{in}})
$$

where $F_{\text{latent}}$ is the first $n_\text{latent}$ columns of $F$ (the columns
corresponding to latent factors). $G$ folds the anchoring scales into the transition so
that the covariance update works directly in the un-anchored (internal) scale.

The predicted covariance satisfies

$$
\hat{P} = G\, P\, G^\top + Q
$$

where $Q = \operatorname{diag}(\sigma / s^{\text{out}})^2$ and $\sigma$ is the vector of
shock standard deviations. In square-root form, the upper Cholesky factor $\hat{R}$ of
$\hat{P}$ is obtained via a single QR decomposition of the stacked matrix

$$
S = \begin{bmatrix} R\, G^\top \\ \operatorname{diag}(\sigma / s^{\text{out}})
    \end{bmatrix}
$$

which has shape $(2n) \times n$. The upper-triangular $R$-factor of $S$ (its first $n$
rows) gives $\hat{R}$.

## Observed factors

Observed factors (e.g. investment measures whose values are known from data) appear as
columns in $F$ and therefore influence the predicted mean through the matrix--vector
product. However, they carry no uncertainty: their columns are excluded from the
covariance propagation. This is why $G$ uses only the first $n_\text{latent}$ columns of
$F$ rather than the full matrix.

## Practical impact

Benchmarks on a 4-factor linear model (`health-cognition`,
`no_feedback_to_investments_linear`, 8 GiB GPU) show a modest ~6 % speed-up on GPU (8.4
vs 8.9 s per optimizer iteration) and negligible difference on CPU. The speed gain is
small because with only 4 latent factors the unscented transform generates just 9 sigma
points — a trivially cheap operation on modern hardware.

The memory reduction is the more significant benefit. Under the same conditions the
unscented path ran out of GPU memory when only ~5 GiB was free, while the linear path
ran without issues. For models with more latent factors both advantages grow: the
sigma-point count scales as $2n + 1$ and the QR matrix shrinks from $(3n + 1) \times n$
to $(2n) \times n$.
