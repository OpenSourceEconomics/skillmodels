.. _names_and_concepts:


******************
Names and concepts
******************

This section contains an overview of frequently used variable names and
concepts. It's not necessary to read this section if you are only interested in
using the code, but you might want to skim it if you are interested in what the
code actually does or plan to adapt it to your use case.

Variables related to dimensions
*******************************

    * **nfac**: number of latent factors
    * **nperiods**: number of periods
    * **nstages**: number of stages
    * **nemf**: number of elements in the mixture-of-normals distribution of
      the latent factors
    * **nobs**: number of observations
    * **nupdates**: total number of Kalman updates, including those for
      anchoring equations

.. _params_and_quants:

Params and the quantities that depend on it
*******************************************

params is a vector with all estimated parameters of the model. To evaluate the
likelihood function, params has to be parsed into several quantities (Names
follow conventions of Kalman filtering literature where possible):

    * **deltas**: list with one matrix per period with the estimated parameters
      related to control variables in the measurement and anchoring
      equations, including constants
    * **H**: Matrix with factor loadings for the measurement and anchoring equations,
      i.e. for all Kalman updates. It has the shape [nupdates, nfac].
    * **R**: Vector of length nupdates with the variances of the measurement equations
    * **Q**: List of [nfac x nfac] matrices for each stage with the variances of the
      transition equations on the diagonals.
    * **X_zero**: Numpy array of [nemf * nind, nfac] with the initial means of the
      latent factors
    * **P_zero**: Numpy array of [nemf * nind, nfac, nfac] with the initial covariances
      of the latent factors
    * **W_zero**: array of shape [nemf, nind] with the initial weights of the mixture
      elements in the factor distribution
    * **trans_coeffs**: list of arrays with transition equation coefficients for
      each latent factor.

For efficiency reasons all of these quantities are only created once and then
overwritten  with new parameters in each iteration of the likelihood
maximization.
