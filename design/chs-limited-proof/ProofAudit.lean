module

import CHS

/-! Inspect the trusted axioms used by every theorem in this project. -/

-- ErrorPropagation.lean
#print axioms CHS.ErrorPropagation.abs_log_sub_log_le_div
#print axioms CHS.ErrorPropagation.logLikelihood_error
#print axioms CHS.ErrorPropagation.logLikelihood_error_of_score_error
#print axioms CHS.ErrorPropagation.errorEnvelope_nonneg
#print axioms CHS.ErrorPropagation.error_le_envelope
#print axioms CHS.ErrorPropagation.error_le_envelope_with_prediction
#print axioms CHS.ErrorPropagation.logLikelihood_error_with_prediction
#print axioms CHS.ErrorPropagation.geometricAccumulation_nonneg
#print axioms CHS.ErrorPropagation.error_le_cubic_envelope
#print axioms CHS.ErrorPropagation.logLikelihood_error_cubic
#print axioms CHS.ErrorPropagation.error_eq_zero_of_zero_defects
#print axioms CHS.ErrorPropagation.logLikelihood_eq_of_zero_defects
#print axioms CHS.ErrorPropagation.normalized_quotient_error
#print axioms CHS.ErrorPropagation.bayesExpectation_error
#print axioms CHS.ErrorPropagation.bayesExpectation_error_nine

-- GaussianProjection.lean
#print axioms CHSLimited.integral_quadratic
#print axioms CHSLimited.MomentMatch.integral_quadratic_eq
#print axioms CHSLimited.MomentMatch.integral_centered_sq_eq
#print axioms CHSLimited.MomentMatch.integral_affine_eq
#print axioms CHSLimited.MomentMatch.integral_affine_sq_eq
#print axioms CHSLimited.gaussianProjection_mean
#print axioms CHSLimited.gaussianProjection_variance
#print axioms CHSLimited.gaussianProjection_memLp_two
#print axioms CHSLimited.gaussianProjection_second
#print axioms CHSLimited.gaussianProjection_momentMatch
#print axioms CHSLimited.gaussianProjection_centered_second
#print axioms CHSLimited.gaussianProjection_integral_quadratic
#print axioms CHSLimited.gaussianProjection_gaussianReal
#print axioms CHSLimited.gaussianProjection_idempotent
#print axioms CHSLimited.gaussianProjection_integrable_abs_cube
#print axioms CHSLimited.memLp_id_map
#print axioms CHSLimited.gaussianProjection_momentMatch_map
#print axioms CHSLimited.gaussianProjection_mean_map
#print axioms CHSLimited.gaussianProjection_second_map
#print axioms CHSLimited.gaussianProjection_integral_quadratic_map
#print axioms CHSLimited.gaussianProjection_isProbabilityMeasure

-- GaussianScore.lean
#print axioms CHS.GaussianScore.momentScore_excess
#print axioms CHS.GaussianScore.variance_mismatch_nonneg
#print axioms CHS.GaussianScore.momentScore_minimized
#print axioms CHS.GaussianScore.variance_mismatch_pos
#print axioms CHS.GaussianScore.momentScore_eq_min_iff
#print axioms CHS.GaussianScore.pointScore_integrable
#print axioms CHS.GaussianScore.integral_sq_error
#print axioms CHS.GaussianScore.integral_pointScore
#print axioms CHS.GaussianScore.expectedScore_unique_minimizer

-- LimitedApproximation.lean
#print axioms CHSLimited.MomentMatch.smooth_log_score_bound
#print axioms CHSLimited.MomentMatch.smooth_bayes_bound

-- Mixture.lean
#print axioms CHS.Mixture.weighted_sum_error
#print axioms CHS.Mixture.weighted_sum_error_uniform
#print axioms CHS.Mixture.posteriorWeight_nonneg
#print axioms CHS.Mixture.posteriorWeight_sum
#print axioms CHS.Mixture.posteriorWeights_are_probabilities
#print axioms CHS.Mixture.posterior_weighted_error
#print axioms CHS.Mixture.current_update_unchanged_of_component_scores_equal

-- UpdateAlgebra.lean
#print axioms CHS.UpdateAlgebra.directionalUpdate_le_prior
#print axioms CHS.UpdateAlgebra.directionalUpdate_lower_bound
#print axioms CHS.UpdateAlgebra.directionalUpdate_pos
#print axioms CHS.UpdateAlgebra.directionalUpdate_zero_cross
#print axioms CHS.UpdateAlgebra.directionalUpdate_zero_contraction
#print axioms CHS.UpdateAlgebra.directionalUpdate_zero_loading_variance
#print axioms CHS.UpdateAlgebra.residual_variance_identity
#print axioms CHS.UpdateAlgebra.bounded_coefficient_identity
#print axioms CHS.UpdateAlgebra.bounded_factor_quadratic_identity
#print axioms CHS.UpdateAlgebra.squaredNorm_nonneg
#print axioms CHS.UpdateAlgebra.coordinate_cauchy
#print axioms CHS.UpdateAlgebra.squaredNorm_rankOne_factor
#print axioms CHS.UpdateAlgebra.coordinate_update_bounds
#print axioms CHS.UpdateAlgebra.coordinate_update_pos
#print axioms CHS.UpdateAlgebra.coordinate_update_bounds_of_retained_variance
#print axioms CHS.UpdateAlgebra.coordinate_update_zero_loading
#print axioms CHS.UpdateAlgebra.affine_residual_mean
#print axioms CHS.UpdateAlgebra.affine_residual_variance
#print axioms CHS.UpdateAlgebra.selected_affine_moments

-- WeakError.lean
#print axioms CHSLimited.integrable_quadratic
#print axioms CHSLimited.MomentMatch.remainder_bound
#print axioms CHSLimited.MomentMatch.cubic_remainder_bound
#print axioms CHSLimited.taylorWithinEval_two_eq
#print axioms CHSLimited.quadraticTaylor_remainder_bound
#print axioms CHSLimited.MomentMatch.smooth_scaled_bound
#print axioms CHSLimited.MomentMatch.smooth_scaled_centered_bound
#print axioms CHSLimited.integrable_of_cubic_remainder
#print axioms CHSLimited.smooth_scaled_integrable
#print axioms CHSLimited.gaussianProjection_smooth_scaled_bound

