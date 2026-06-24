"""AF probit/Tobit measurement support: hand-computed likelihood acceptance tests.

With fixed Halton nodes, replacing a Gaussian measure by a probit (or Tobit)
measure must equal a manually node-weighted likelihood -- there is no extra
approximation beyond the existing Halton integration. These tests pin that against
plain-numpy/scipy references.
"""

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from skillmodels.af.likelihood import af_per_obs_loglike_initial
from skillmodels.common.measurement_models import MeasurementFamily

_INF = float("inf")


def test_initial_probit_measure_matches_node_weighted_reference() -> None:
    """A binary probit measure at period 0 equals a hand node-weighted likelihood.

    Single latent factor, single mixture component, no observed factors (so the
    unconditional integrand runs). One Gaussian measure plus one probit measure;
    the probit contributes ``log Phi((2y-1) eta)`` at each Halton node.
    """
    n_factors = 1
    n_latent = 1
    n_components = 1
    n_measures = 2
    n_controls = 1

    mu = 0.5
    chol = 1.1
    control_params = [0.1, -0.2]
    loadings = [1.0, 0.8]
    meas_sds = [0.5, 1.0]  # probit sd is ignored by the kernel

    params = jnp.array(
        [1.0, mu, chol, *control_params, *loadings, *meas_sds],
    )
    loading_mask = jnp.array([[True], [True]])

    n_obs = 4
    rng = np.random.default_rng(7)
    y_gauss = rng.normal(0, 1, n_obs)
    y_probit = (rng.random(n_obs) < 0.5).astype(float)
    measurements = jnp.asarray(np.column_stack([y_gauss, y_probit]))
    controls = jnp.asarray(rng.normal(0, 1, (n_obs, n_controls)))

    raw_nodes = np.array([-1.5, -0.5, 0.5, 1.5])
    node_w = np.exp(-0.5 * raw_nodes**2)
    node_w = node_w / node_w.sum()
    nodes = jnp.asarray(raw_nodes.reshape(-1, 1))
    weights = jnp.asarray(node_w)

    families = jnp.array(
        [int(MeasurementFamily.GAUSSIAN), int(MeasurementFamily.PROBIT)]
    )
    lowers = jnp.array([-_INF, -_INF])
    uppers = jnp.array([_INF, _INF])

    per_obs = np.asarray(
        af_per_obs_loglike_initial(
            params,
            n_factors=n_factors,
            n_mixture_components=n_components,
            n_measures=n_measures,
            n_controls=n_controls,
            measurements=measurements,
            controls=controls,
            loading_mask=loading_mask,
            nodes=nodes,
            weights=weights,
            stability_floor=0.0,
            n_latent_factors=n_latent,
            measurement_families=families,
            measurement_lowers=lowers,
            measurement_uppers=uppers,
        )
    )

    control_arr = np.array(control_params).reshape(n_measures, n_controls)
    load = np.array(loadings)
    raw = np.asarray(raw_nodes)
    expected = np.empty(n_obs)
    for i in range(n_obs):
        ctrl_i = np.asarray(controls[i])
        eta_const = control_arr @ ctrl_i  # (n_measures,)
        node_contrib = np.empty(len(raw))
        for q, z in enumerate(raw):
            theta = mu + chol * z
            eta = eta_const + load * theta
            logp_g = norm.logpdf(y_gauss[i], loc=eta[0], scale=meas_sds[0])
            logp_p = norm.logcdf((2.0 * y_probit[i] - 1.0) * eta[1])
            node_contrib[q] = np.exp(logp_g + logp_p)
        expected[i] = np.log(np.dot(node_w, node_contrib))

    np.testing.assert_allclose(per_obs, expected, rtol=1e-6, atol=1e-9)


def test_initial_all_gaussian_families_match_default_path() -> None:
    """Passing explicit all-Gaussian families equals omitting them (parity)."""
    n_measures = 2
    params = jnp.array([1.0, 0.3, 1.0, 0.0, 0.0, 1.0, 0.7, 0.5, 0.6])
    loading_mask = jnp.array([[True], [True]])
    rng = np.random.default_rng(1)
    measurements = jnp.asarray(rng.normal(0, 1, (5, n_measures)))
    controls = jnp.asarray(np.ones((5, 1)))
    raw = np.array([-1.0, 0.0, 1.0])
    w = np.exp(-0.5 * raw**2)
    w = w / w.sum()
    nodes = jnp.asarray(raw.reshape(-1, 1))
    weights = jnp.asarray(w)

    def _call(families, lowers, uppers):
        return np.asarray(
            af_per_obs_loglike_initial(
                params,
                n_factors=1,
                n_mixture_components=1,
                n_measures=n_measures,
                n_controls=1,
                measurements=measurements,
                controls=controls,
                loading_mask=loading_mask,
                nodes=nodes,
                weights=weights,
                stability_floor=0.0,
                n_latent_factors=1,
                measurement_families=families,
                measurement_lowers=lowers,
                measurement_uppers=uppers,
            )
        )

    default = _call(None, None, None)
    explicit = _call(
        jnp.array([int(MeasurementFamily.GAUSSIAN)] * n_measures),
        jnp.array([-_INF] * n_measures),
        jnp.array([_INF] * n_measures),
    )
    np.testing.assert_allclose(default, explicit, rtol=1e-12, atol=1e-12)
