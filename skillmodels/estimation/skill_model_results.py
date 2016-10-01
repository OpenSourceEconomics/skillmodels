from statsmodels.base.model import GenericLikelihoodModelResults
from statsmodels.tools.decorators import resettable_cache, cache_readonly
import numpy as np


class NotApplicableError(Exception):
    pass


class SkillModelResults(GenericLikelihoodModelResults):
    def __init__(self, model, mlefit, optimize_dict=None):
        self.model = model
        self.estimator = model.estimator
        self.optimize_dict = optimize_dict
        self.nobs = model.nobs
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = resettable_cache()
        self.__dict__.update(mlefit.__dict__)
        self.param_names = model.param_names(params_type='long')

    @cache_readonly
    def aic(self):
        if self.estimator == 'chs':
            return super(SkillModelResults, self).aic
        else:
            raise NotApplicableError(
                'aic only works for likelihood based models.')

    @cache_readonly
    def llf(self):
        if self.estimator == 'chs':
            return self.optimize_dict['log_lh_value']
        else:
            raise NotApplicableError(
                'If the wa estimator was used there is no likelihood value.')

    @cache_readonly
    def df_modelwc(self):
        return self.df_model

    @cache_readonly
    def bic(self):
        raise NotImplementedError

    @cache_readonly
    def covbs(self):
        """Return boostrap covariance matrix.

        Why is this handled differently than other cov matrices in Statsmodels?

        """
        return self.model.bootstrap_cov_matrix(self.params)

    @cache_readonly
    def bsebs(self):
        """Return bootstrap standard errors.

        Why is this handled differently than other cov matrices in Statsmodels?

        """
        return np.sqrt(np.diag(self.covbs))

    def bootstrap(self):
        """A dictionary with all bootstrapped statistics.

        Currently implemented are:
            * covariance matrix
            * standard_errors
            * mean
            * conf_int

        I plan to add p_values.
        """
        bs_dict = {}
        bs_dict['covariance_matrix'] = self.covbs
        bs_dict['standard_errors'] = self.bsebs
        bs_dict['mean'] = self.model.bootstrap_mean(self.params)
        bs_dict['conf_int'] = self.model.bootstrap_conf_int(self.params)

        return bs_dict

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        raise NotImplementedError(
            'The summary method is not yet implemented for SkillModelResults')

    def get_nlfun(self, fun):
        # in the super class this is a do-nothing function
        # for safety I raise an error here.
        raise NotImplementedError

    def conf_int(self):
        if self.model.standard_error_method == 'bootstrap':
            return self.model.bootstrap_conf_int(self.params)
        else:
            return super(SkillModelResults, self).conf_int()

    def save(self, fname, remove_data=False):
        raise NotImplementedError(
            'A save method is not yet implemented for SkillModelResults')

    @classmethod
    def load(cls, fname):
        raise NotImplementedError(
            'A load method is not yet implemented for SkillModelResults')

    def remove_data(self):
        raise NotImplementedError(
            'A remove_data method is not yet implemented for '
            'SkillModelResults.')

    def predict(self, exog=None, transform=True, *args, **kwargs):
        raise NotImplementedError(
            'A predict method is not yet implemented for SkillModelResults')





