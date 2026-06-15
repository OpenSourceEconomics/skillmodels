"""Construction of the control-function (cf) nodes for the CHS transition DAG.

The first-class control-function correction adds three kinds of synthetic node
to the per-period transition DAG built in `process_model._get_transition_info`:

1. a deterministic, contemporaneous first-stage prediction
   `E[ln I_t | theta_t, Y_t]` for the endogenous investment factor,
2. the residual `cf_t = ln I_t - E[ln I_t | theta_t, Y_t]`, and
3. for each target production factor, the additive `sum_k kappa_k * cf_term_k`
   term grafted onto the factor's transition output.

All nodes operate on the same period-`t` anchored sigma-point `states` row that
the production transitions already consume, which is what makes `cf` a genuine
same-period residual. The functions here are pure builders returning the node
callables; the wiring into the DAG lives in `process_model`.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence

import jax.numpy as jnp
from jax import Array


def build_prediction_node(
    beta_key: str,
    predictor_positions: Sequence[int],
) -> Callable[[Array, Mapping[str, Array]], Array]:
    """Return the contemporaneous first-stage prediction node.

    The node reads the period-`t` anchored `states` row and the reserved
    first-stage coefficient vector `params[beta_key]` (ordered as the
    predictors then a trailing constant) and returns `x @ betas`.

    Args:
        beta_key: Reserved transition-coeffs key holding the first-stage betas.
        predictor_positions: Positions of the predictor/instrument factors in
            the `states` vector, in the same order as the first-stage betas.

    Returns:
        A node `(states, params) -> prediction`.

    """
    pos = jnp.array(tuple(predictor_positions), dtype=int)

    def prediction(states: Array, params: Mapping[str, Array]) -> Array:
        betas = params[beta_key]
        x = jnp.concatenate([states[pos], jnp.array([1.0])])
        return jnp.dot(x, betas)

    return prediction


def compute_investment_residual_sds(
    investment: Array,
    predictors: Array,
    betas: Array,
) -> Array:
    """Report SD(eta_I) per period: the first-stage residual's cross-sectional SD.

    A derived diagnostic, not a free MLE parameter. `eta_{I,t} = ln I_t -
    E[ln I_t | theta_t, Y_t]` and this returns its standard deviation over
    observations for each period, mirroring the AMN first-stage residual SD. The
    caller supplies the (filtered or simulated) investment values and predictor
    panel and the estimated first-stage coefficients.

    Args:
        investment: Shape `(n_obs, n_periods)`. The investment factor values.
        predictors: Shape `(n_obs, n_periods, n_predictors)`. The first-stage
            predictors and instruments, in the coefficient order.
        betas: Shape `(n_periods, n_predictors + 1)`. First-stage coefficients
            with the constant last.

    Returns:
        Shape `(n_periods,)` with SD(eta_I) for each period.

    """
    ones = jnp.ones((*predictors.shape[:2], 1))
    design = jnp.concatenate([predictors, ones], axis=-1)
    prediction = jnp.einsum("opk,pk->op", design, betas)
    return jnp.std(investment - prediction, axis=0)


def build_cf_node(inv_pos: int) -> Callable[[Array, Array], Array]:
    """Return the residual node `cf = ln I_t - prediction`.

    Args:
        inv_pos: Position of the investment factor in the `states` vector.

    Returns:
        A node `(states, prediction) -> cf`.

    """

    def cf(states: Array, prediction: Array) -> Array:
        return states[inv_pos] - prediction

    return cf


def build_kappa_term_evaluators(
    kappa_terms: Sequence[str],
    factor_positions: Mapping[str, int],
) -> list[Callable[[Array, Array], Array]]:
    """Return one evaluator per kappa term.

    Each evaluator maps `(cf, states) -> term value`: `"cf"` -> `cf`,
    `"cf ** 2"` -> `cf ** 2`, and `"cf * <factor>"` -> `cf * states[pos]`.

    Args:
        kappa_terms: The cf regressor names for one target factor.
        factor_positions: Positions of factors in the `states` vector, used to
            resolve interaction terms.

    Returns:
        Evaluators aligned with `kappa_terms`.

    Raises:
        ValueError: If a kappa term is not one of the supported forms.

    """
    return [_make_monomial_evaluator(term, factor_positions) for term in kappa_terms]


def _parse_kappa_term(
    term: str,
    factor_positions: Mapping[str, int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Parse a kappa monomial into its cf power and `(position, power)` factors.

    A kappa term is a monomial `cf ** a * factor_1 ** b_1 * ...` written with the
    same spacing as the built-in transition parameter names (`" * "` between
    atoms, `" ** "` for powers, a bare name for power 1). Returns the cf power
    (which must be at least one) and the `(states position, power)` pairs.
    """
    cf_power = 0
    factor_powers: list[tuple[int, int]] = []
    for atom in term.split(" * "):
        if atom == "cf":
            cf_power += 1
        elif atom.startswith("cf ** "):
            cf_power += int(atom.removeprefix("cf ** "))
        elif " ** " in atom:
            factor, power = atom.split(" ** ")
            factor_powers.append((factor_positions[factor], int(power)))
        else:
            factor_powers.append((factor_positions[atom], 1))
    if cf_power == 0:
        msg = (
            f"Kappa term {term!r} must include cf (e.g. 'cf', 'cf ** 2', "
            "'cf * factor', 'cf ** 2 * factor_1 * factor_2')."
        )
        raise ValueError(msg)
    return cf_power, tuple(factor_powers)


def _make_monomial_evaluator(
    term: str,
    factor_positions: Mapping[str, int],
) -> Callable[[Array, Array], Array]:
    cf_power, factor_powers = _parse_kappa_term(term, factor_positions)

    def evaluator(cf: Array, states: Array) -> Array:
        result = cf**cf_power
        for position, power in factor_powers:
            result = result * states[position] ** power
        return result

    return evaluator


def generate_kappa_terms(
    factors: Sequence[str],
    max_degree: int,
    max_cf_power: int | None = None,
) -> tuple[str, ...]:
    """Generate the complete cf-interaction basis up to a total degree.

    Every monomial `cf ** a * prod_i factor_i ** b_i` with `a >= 1`, `b_i >= 0`
    and `a + sum_i b_i <= max_degree` (optionally capping the cf power at
    `max_cf_power`). At `max_degree=1` this is just `("cf",)`; at `max_degree=2`
    over two factors it is the translog set `cf, cf * f1, cf * f2, cf ** 2`.
    Pass the result as a target's `kappa_terms` and pin unwanted coefficients to
    zero with an optimagic constraint.

    Args:
        factors: The state factors that interact with cf.
        max_degree: Maximum total degree of a monomial (cf power plus factor
            powers).
        max_cf_power: Optional cap on the cf power. Defaults to `max_degree`.

    Returns:
        The kappa term names, in ascending degree order.

    """
    factors = tuple(factors)
    cap = max_degree if max_cf_power is None else min(max_cf_power, max_degree)
    terms: list[str] = []
    for cf_power in range(1, cap + 1):
        for factor_powers in _bounded_power_tuples(len(factors), max_degree - cf_power):
            terms.append(_monomial_name(cf_power, factor_powers, factors))
    return tuple(terms)


def _bounded_power_tuples(n_factors: int, total: int) -> Iterator[tuple[int, ...]]:
    """Yield non-negative integer tuples of length `n_factors` summing to <= total."""
    if n_factors == 0:
        yield ()
        return
    for first in range(total + 1):
        for rest in _bounded_power_tuples(n_factors - 1, total - first):
            yield (first, *rest)


def _monomial_name(
    cf_power: int,
    factor_powers: Sequence[int],
    factors: Sequence[str],
) -> str:
    parts = ["cf" if cf_power == 1 else f"cf ** {cf_power}"]
    for factor, power in zip(factors, factor_powers, strict=True):
        if power == 1:
            parts.append(factor)
        elif power >= 2:
            parts.append(f"{factor} ** {power}")
    return " * ".join(parts)


def build_kappa_addition_node(
    kappa_key: str,
    kappa_evaluators: Sequence[Callable[[Array, Array], Array]],
) -> Callable[[Array, Mapping[str, Array], Array, Array], Array]:
    """Add `sum_k kappa_k * cf_term_k` to a target factor's base output.

    Kappa coefficients live in their own transition-coeffs key (a separate
    `kappa` params category), so the base production node is left completely
    untouched: its arguments, parameter layout, and DAG dependencies are
    unchanged, which keeps both built-in (positional, constant-last) and custom
    (named-factor) transitions working. The kappa contribution is added to the
    scalar base output, outside any translog/CES aggregator.

    Args:
        kappa_key: The target's key into the transition-coeffs dict holding its
            period-sliced kappa coefficients (aligned with `kappa_evaluators`).
        kappa_evaluators: One evaluator per kappa term (see
            `build_kappa_term_evaluators`).

    Returns:
        A node `(base_value, params, cf, states) -> next state`. The first
        argument is the base factor transition's output; the caller renames it
        to the base node's name when wiring the DAG.

    """

    def add_kappa(
        base_value: Array,
        params: Mapping[str, Array],
        cf: Array,
        states: Array,
    ) -> Array:
        kappa = params[kappa_key]
        result = base_value
        for coefficient, evaluator in zip(kappa, kappa_evaluators, strict=True):
            result = result + coefficient * evaluator(cf, states)
        return result

    return add_kappa
