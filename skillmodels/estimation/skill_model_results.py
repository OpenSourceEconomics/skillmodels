from statsmodels.base.model import GenericLikelihoodModelResults
from statsmodels.tools.decorators import resettable_cache, cache_readonly


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
            return super(SkillModelResults, self).aic()
        else:
            raise NotApplicableError(
                'aic only works for likelihood based models.')

    @cache_readonly
    def df_modelwc(self):
        return self.df_model

    @cache_readonly
    def bic(self):
        raise NotImplementedError






    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        raise NotImplementedError(
            'The summary method is not yet implemented for CHSModelResults')


    @cache_readonly
    def llf(self):
        if self.estimator == 'chs':
            return self.optimize_dict['log_lh_value']
        else:
            raise NotApplicableError(
                'If the wa estimator was used there is no likelihood value.')

    @cache_readonly
    def score_obsv(self):
        if self.estimator == 'wa':
            raise NotApplicableError
        raise NotImplementedError

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        # TODO: write this function. should be a call to model.bootstrap
        # or a lookup if this is the standard_error_method
        raise NotImplementedError(
            'Bootstrap is not yet implemented for CHSModelResults')

    def get_nlfun(self, fun):
        raise NotImplementedError

    def save(self, fname, remove_data=False):
        raise NotImplementedError(
            'A save method is not yet implemented for CHSModelResults')

    @classmethod
    def load(cls, fname):
        raise NotImplementedError(
            'A load method is not yet implemented for CHSModelResults')

    def remove_data(self):
        raise NotImplementedError(
            'A remove_data method is not yet implemented for CHSModelResults')

    def predict(self, exog=None, transform=True, *args, **kwargs):
        raise NotImplementedError(
            'A predict method is not yet implemented for CHSModelResults')




    @cache_readonly
    def hessv(self):
        raise NotImplementedError

    @cache_readonly
    def covjac(self):
        raise NotImplementedError

    @cache_readonly
    def covjhj(self):
        raise NotImplementedError

    @cache_readonly
    def bsejhj(self):
        raise NotImplementedError

    @cache_readonly
    def bsejac(self):
        raise NotImplementedError



